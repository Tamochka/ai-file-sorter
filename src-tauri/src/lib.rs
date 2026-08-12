use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, fs, path::{Component, Path, PathBuf}};
use uuid::Uuid;
use walkdir::WalkDir;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AiSettings { provider: String, base_url: String, model: String, api_key: String, cloud_consent: bool }
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SortSettings { mode: String, custom_prompt: String, text_limit: usize, total_limit: usize }
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct PlanItem { id: String, source: String, relative_path: String, target: String, category: String, explanation: String, confidence: f32, included: bool, warning: Option<String> }
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AnalysisResult { items: Vec<PlanItem>, total_files: usize, estimated_chars: usize, warnings: Vec<String> }
#[derive(Debug, Serialize, Deserialize)]
struct MoveRecord { from: String, to: String }
#[derive(Debug, Deserialize)]
struct AiDecision { id: String, category: String, explanation: Option<String>, confidence: Option<f32> }
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AiFileContext { id: String, path: String, extension: String, size_bytes: u64, created_at: Option<String>, modified_at: Option<String>, suggested_category: String, content_extract: Option<String>, content_status: String }
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelList { models: Vec<String>, active_model: Option<String> }

const SORTED_DIR: &str = "AI Sorted";
const HISTORY_FILE: &str = ".ai-file-sorter-last-operation.json";

#[tauri::command]
fn analyze_folder(folder: String, ai: AiSettings, sort: SortSettings) -> Result<AnalysisResult, String> {
  let root = canonical_root(&folder)?;
  if ai.model.trim().is_empty() { return Err("Укажите имя модели".into()); }
  let is_local = ai.provider == "lmstudio" || ai.provider == "ollama" || ai.base_url.contains("localhost") || ai.base_url.contains("127.0.0.1");
  if !is_local && !ai.cloud_consent { return Err("Нужно подтвердить передачу данных в облако".into()); }
  let mut total_chars = 0usize; let mut items = Vec::new(); let mut contexts = Vec::new(); let mut warnings = Vec::new();
  for entry in WalkDir::new(&root).into_iter().filter_entry(scan_entry).filter_map(Result::ok).filter(|e| e.file_type().is_file()) {
    let path = entry.path(); let relative = path.strip_prefix(&root).map_err(|_| "Не удалось вычислить относительный путь")?;
    if skip_file(path) { continue; }
    let metadata = match fs::metadata(path) { Ok(data) => data, Err(_) => { warnings.push(format!("Нет доступа к {}", relative.display())); continue; } };
    let ext = path.extension().and_then(|x| x.to_str()).unwrap_or("").to_lowercase();
    let remaining = sort.total_limit.saturating_sub(total_chars);
    let (content_extract, content_status) = read_text_preview(path, &ext, sort.text_limit.min(remaining));
    total_chars = total_chars.saturating_add(content_extract.as_ref().map_or(0, |text| text.chars().count()));
    let (category, confidence, explanation) = classify(relative, &ext, &sort);
    let date = metadata.modified().ok().map(|t| DateTime::<Local>::from(t).format("%Y/%m").to_string()).unwrap_or_else(|| "Без даты".into());
    let kind = file_kind(&ext);
    let target = Path::new(SORTED_DIR).join(safe_name(&category)).join(kind).join(date).join(relative);
    let id = Uuid::new_v4().to_string();
    contexts.push(AiFileContext { id: id.clone(), path: relative.to_string_lossy().into_owned(), extension: ext.clone(), size_bytes: metadata.len(), created_at: format_file_time(metadata.created().ok()), modified_at: format_file_time(metadata.modified().ok()), suggested_category: category.clone(), content_extract, content_status });
    items.push(PlanItem { id, source: path.to_string_lossy().into_owned(), relative_path: relative.to_string_lossy().into_owned(), target: target.to_string_lossy().into_owned(), category, explanation, confidence, included: true, warning: unsupported_warning(&ext) });
  }
  if sort.mode == "custom" && sort.custom_prompt.trim().is_empty() { warnings.push("Кастомный режим без инструкции использовал стандартные категории.".into()); }
  if let Err(error) = refine_with_model(&ai, &sort, &mut items, &contexts) { warnings.push(format!("Модель не уточнила план: {error}. Использована локальная оценка по метаданным.")); }
  if total_chars >= sort.total_limit { warnings.push("Достигнут общий лимит текста. Часть файлов будет оценена по имени и метаданным.".into()); }
  Ok(AnalysisResult { total_files: items.len(), estimated_chars: total_chars, items, warnings })
}

#[tauri::command]
fn apply_sort(folder: String, items: Vec<PlanItem>) -> Result<usize, String> {
  let root = canonical_root(&folder)?; let mut records = Vec::new(); let mut seen = HashSet::new();
  for item in items {
    let source = canonical_inside(&root, Path::new(&item.source))?;
    let destination = safe_destination(&root, &item.target)?;
    if source == destination { continue; }
    if !seen.insert(destination.clone()) { return Err(format!("Повторяющийся целевой путь: {}", destination.display())); }
    let destination = conflict_free(&destination);
    if let Some(parent) = destination.parent() { fs::create_dir_all(parent).map_err(io_error)?; }
    fs::rename(&source, &destination).map_err(io_error)?;
    records.push(MoveRecord { from: source.to_string_lossy().into_owned(), to: destination.to_string_lossy().into_owned() });
  }
  let history = root.join(HISTORY_FILE); fs::write(history, serde_json::to_vec_pretty(&records).map_err(|e| e.to_string())?).map_err(io_error)?;
  Ok(records.len())
}

#[tauri::command]
fn undo_last_sort(folder: String) -> Result<usize, String> {
  let root = canonical_root(&folder)?; let history = root.join(HISTORY_FILE);
  let raw = fs::read(&history).map_err(|_| "Нет операции для отмены".to_string())?;
  let records: Vec<MoveRecord> = serde_json::from_slice(&raw).map_err(|_| "Журнал операции повреждён".to_string())?;
  let mut restored = 0;
  for record in records.into_iter().rev() {
    let current = canonical_inside(&root, Path::new(&record.to))?;
    let original = safe_destination(&root, &record.from)?; let original = conflict_free(&original);
    if let Some(parent) = original.parent() { fs::create_dir_all(parent).map_err(io_error)?; }
    fs::rename(current, original).map_err(io_error)?; restored += 1;
  }
  fs::remove_file(history).map_err(io_error)?; Ok(restored)
}

#[tauri::command]
fn test_connection(ai: AiSettings) -> Result<String, String> {
  if ai.base_url.trim().is_empty() { return Err("Укажите базовый URL".into()); }
  let url = format!("{}/models", ai.base_url.trim_end_matches('/'));
  let mut request = ureq::get(&url).timeout(std::time::Duration::from_secs(8));
  if !ai.api_key.trim().is_empty() { request = request.set("Authorization", &format!("Bearer {}", ai.api_key)); }
  request.call().map(|_| format!("Подключение успешно: {}", ai.base_url)).map_err(|e| format!("Сервис не ответил: {e}"))
}

#[tauri::command]
fn list_models(ai: AiSettings) -> Result<ModelList, String> {
  if ai.base_url.trim().is_empty() { return Err("Укажите базовый URL".into()); }
  let url = if ai.provider == "lmstudio" {
    format!("{}/api/v1/models", ai.base_url.trim_end_matches('/').trim_end_matches("/v1"))
  } else if ai.provider == "ollama" {
    format!("{}/api/tags", ai.base_url.trim_end_matches('/').trim_end_matches("/v1"))
  } else { format!("{}/models", ai.base_url.trim_end_matches('/')) };
  let mut request = ureq::get(&url).timeout(std::time::Duration::from_secs(12));
  if !ai.api_key.trim().is_empty() { request = request.set("Authorization", &format!("Bearer {}", ai.api_key)); }
  let value: serde_json::Value = request.call().map_err(|e| format!("Не удалось получить модели: {e}"))?.into_json().map_err(|e| e.to_string())?;
  let source = if ai.provider == "lmstudio" || ai.provider == "ollama" { value.get("models").and_then(|v| v.as_array()) } else { value.get("data").and_then(|v| v.as_array()) }.ok_or("Сервис вернул список моделей в неизвестном формате")?;
  if ai.provider == "lmstudio" {
    let active_model = source.iter().filter(|model| model.get("type").and_then(|kind| kind.as_str()) == Some("llm")).find_map(|model| model.get("loaded_instances").and_then(|instances| instances.as_array()).and_then(|instances| instances.first()).and_then(|instance| instance.get("id")).and_then(|id| id.as_str()).map(str::to_owned));
    let mut models: Vec<String> = source.iter().filter(|model| model.get("type").and_then(|kind| kind.as_str()) == Some("llm")).filter_map(|model| model.get("key").and_then(|key| key.as_str()).map(str::to_owned)).collect();
    if let Some(active) = &active_model { models.retain(|model| model != active); models.insert(0, active.clone()); }
    if models.is_empty() { return Err("LM Studio не сообщил доступных языковых моделей".into()); }
    return Ok(ModelList { models, active_model });
  }
  let mut models: Vec<String> = source.iter().filter_map(|model| model.get(if ai.provider == "ollama" { "name" } else { "id" }).and_then(|name| name.as_str()).map(str::to_owned)).collect();
  models.sort(); models.dedup(); if models.is_empty() { return Err("Локальный сервис не сообщил доступных моделей".into()); } Ok(ModelList { models, active_model: None })
}

fn scan_entry(entry: &walkdir::DirEntry) -> bool { if entry.depth() == 0 { return true; } let name = entry.file_name().to_string_lossy(); !(entry.file_type().is_dir() && (name == SORTED_DIR || name.to_ascii_lowercase().ends_with(".app"))) }
fn skip_file(path: &Path) -> bool { matches!(path.file_name().and_then(|name| name.to_str()), Some(".DS_Store") | Some(".localized")) }
fn canonical_root(folder: &str) -> Result<PathBuf, String> { let path = fs::canonicalize(folder).map_err(io_error)?; if !path.is_dir() { return Err("Выбранный путь не является папкой".into()); } Ok(path) }
fn canonical_inside(root: &Path, candidate: &Path) -> Result<PathBuf, String> { let path = fs::canonicalize(candidate).map_err(io_error)?; if !path.starts_with(root) { return Err("Путь выходит за пределы выбранной папки".into()); } Ok(path) }
fn safe_destination(root: &Path, relative: &str) -> Result<PathBuf, String> { let p = Path::new(relative); if p.is_absolute() || p.components().any(|c| matches!(c, Component::ParentDir | Component::RootDir | Component::Prefix(_))) { return Err("Недопустимый целевой путь".into()); } if p.components().any(|c| c.as_os_str().to_string_lossy().chars().any(|ch| matches!(ch, ':' | '*' | '?' | '"' | '<' | '>' | '|'))) { return Err("Недопустимые символы в имени папки или файла".into()); } let out = root.join(p); if !out.starts_with(root) { return Err("Путь выходит за пределы выбранной папки".into()); } Ok(out) }
fn conflict_free(path: &Path) -> PathBuf { if !path.exists() { return path.to_path_buf(); } let stem = path.file_stem().and_then(|x| x.to_str()).unwrap_or("file"); let ext = path.extension().and_then(|x| x.to_str()).map(|x| format!(".{x}")).unwrap_or_default(); for i in 2.. { let candidate = path.with_file_name(format!("{stem} ({i}){ext}")); if !candidate.exists() { return candidate; } } unreachable!() }
fn io_error(e: std::io::Error) -> String { e.to_string() }
fn safe_name(value: &str) -> String { let clean: String = value.chars().map(|c| if matches!(c, '/'|'\\'|':'|'*'|'?'|'"'|'<'|'>'|'|') { '_' } else { c }).collect(); if clean.trim().is_empty() { "Прочее".into() } else { clean } }
fn file_kind(ext: &str) -> &'static str { match ext { "dmg"|"pkg"|"mpkg"|"exe"|"msi"|"appimage"|"deb"|"rpm"|"apk"|"iso" => "Установщики", "pdf" => "PDF", "doc"|"docx"|"odt" => "Документы", "xls"|"xlsx"|"csv" => "Таблицы", "txt"|"md"|"rtf" => "Текст", "jpg"|"jpeg"|"png"|"webp"|"heic" => "Изображения", "mp3"|"wav"|"m4a" => "Аудио", "mp4"|"mov"|"mkv" => "Видео", _ => "Прочие файлы" } }
fn unsupported_warning(ext: &str) -> Option<String> { if matches!(ext, "mp3"|"wav"|"m4a"|"mp4"|"mov"|"mkv") { Some("Содержимое аудио/видео в первой версии не анализируется.".into()) } else { None } }
fn format_file_time(time: Option<std::time::SystemTime>) -> Option<String> { time.map(|t| DateTime::<Local>::from(t).to_rfc3339()) }
fn read_text_preview(path: &Path, ext: &str, limit: usize) -> (Option<String>, String) { if !matches!(ext, "txt"|"md"|"csv"|"json"|"xml"|"yaml"|"yml"|"toml"|"html"|"htm"|"log"|"py"|"js"|"ts"|"rs"|"java"|"c"|"cpp"|"h"|"sql") { return (None, "Содержимое этого формата не извлекается; использованы метаданные файла.".into()); } if limit == 0 { return (None, "Достигнут общий лимит текста; использованы метаданные файла.".into()); } match fs::read_to_string(path) { Ok(text) => { let excerpt: String = text.chars().take(limit).collect(); if excerpt.is_empty() { (None, "Текстовый файл пуст; использованы метаданные файла.".into()) } else { (Some(excerpt), "Передан фрагмент текстового содержимого и метаданные файла.".into()) } }, Err(_) => (None, "Не удалось прочитать содержимое; использованы метаданные файла.".into()) } }
fn classify(relative: &Path, ext: &str, sort: &SortSettings) -> (String, f32, String) { let name = relative.to_string_lossy().to_lowercase(); let installer = matches!(ext, "dmg"|"pkg"|"mpkg"|"exe"|"msi"|"appimage"|"deb"|"rpm"|"apk"|"iso") || (matches!(ext, "zip"|"7z") && ["setup", "install", "installer", "latest", "download", "tsetup"].iter().any(|word| name.contains(word))); let category = if sort.mode == "custom" && !sort.custom_prompt.trim().is_empty() { "Кастомная категория" } else if installer { "Загрузчики" } else if name.contains("invoice") || name.contains("счёт") || name.contains("налог") || name.contains("bank") { "Финансы" } else if name.contains("resume") || name.contains("проект") || name.contains("contract") || name.contains("договор") { "Работа" } else if name.contains("course") || name.contains("лекц") || name.contains("study") || name.contains("учеб") { "Учёба" } else if matches!(ext, "jpg"|"jpeg"|"png"|"webp"|"mp4"|"mov"|"mkv"|"mp3") { "Медиа" } else if matches!(ext, "zip"|"rar"|"7z"|"tar"|"gz") { "Архив" } else { "Личное" }; let confidence = if category == "Личное" { 0.45 } else if category == "Загрузчики" { 0.95 } else { 0.72 }; let explanation = if category == "Загрузчики" { "Распознан установочный или загрузочный файл по расширению либо имени." } else { "Первичная локальная оценка по имени, расширению и дате. Подключённая модель уточняет её в следующей итерации." }; (category.into(), confidence, explanation.into()) }

fn refine_with_model(ai: &AiSettings, sort: &SortSettings, items: &mut [PlanItem], contexts: &[AiFileContext]) -> Result<(), String> {
  if items.is_empty() { return Ok(()); }
  let instruction = if sort.mode == "custom" { sort.custom_prompt.as_str() } else { "Используй только категории: Работа, Личное, Финансы, Учёба, Медиа, Архив, Загрузчики, Прочее. Установочные файлы с расширениями DMG, EXE, PKG, MSI и похожими всегда относятся к Загрузчикам." };
  let url = format!("{}/chat/completions", ai.base_url.trim_end_matches('/'));
  for batch in contexts.chunks(10) {
    let prompt = format!("Классифицируй только этот небольшой пакет файлов. Инструкция: {instruction}\nДля каждого файла сначала используй contentExtract, если он есть. Если его нет, анализируй только метаданные: path, extension, sizeBytes, даты и suggestedCategory. Не выдумывай содержимое. Верни ТОЛЬКО JSON-массив объектов {{id, category, explanation, confidence}}. category — короткое безопасное имя папки без / и \\.\nФайлы: {}", serde_json::to_string(batch).map_err(|e| e.to_string())?);
    let body = serde_json::json!({"model":ai.model,"temperature":0,"messages":[{"role":"system","content":"Ты отвечаешь строго валидным JSON без Markdown."},{"role":"user","content":prompt}]});
    let mut request = ureq::post(&url).timeout(std::time::Duration::from_secs(90)).set("Content-Type", "application/json");
    if !ai.api_key.trim().is_empty() { request = request.set("Authorization", &format!("Bearer {}", ai.api_key)); }
    let value: serde_json::Value = request.send_json(body).map_err(|e| e.to_string())?.into_json().map_err(|e| e.to_string())?;
    let content = value.pointer("/choices/0/message/content").and_then(|v| v.as_str()).ok_or("Неподдерживаемый ответ API")?;
    let decisions: Vec<AiDecision> = serde_json::from_str(content.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim()).map_err(|_| "Модель вернула не JSON")?;
    for decision in decisions { if let Some(item) = items.iter_mut().find(|i| i.id == decision.id) { if sort.mode == "standard" && item.category == "Загрузчики" { continue; } let category = safe_name(&decision.category); if category != "Прочее" || decision.category.trim() == "Прочее" { item.category = category.clone(); let previous = Path::new(&item.target); let mut revised = PathBuf::from(SORTED_DIR).join(category); for component in previous.components().skip(2) { revised.push(component.as_os_str()); } item.target = revised.to_string_lossy().into_owned(); } if let Some(explanation) = decision.explanation { item.explanation = explanation; } if let Some(confidence) = decision.confidence { item.confidence = confidence.clamp(0.0, 1.0); } } }
  }
  Ok(())
}

pub fn run() { tauri::Builder::default().plugin(tauri_plugin_dialog::init()).invoke_handler(tauri::generate_handler![analyze_folder, apply_sort, undo_last_sort, test_connection, list_models]).run(tauri::generate_context!()).expect("ошибка запуска Tauri"); }

#[cfg(test)]
mod tests { use super::*; #[test] fn target_rejects_escape() { assert!(safe_destination(Path::new("/tmp/root"), "../out").is_err()); } #[test] fn names_are_sanitized() { assert_eq!(safe_name("A/B: C"), "A_B_ C"); } #[test] fn standard_categories_work() { let sort = SortSettings { mode:"standard".into(), custom_prompt:"".into(), text_limit:1, total_limit:1 }; assert_eq!(classify(Path::new("tax_invoice.pdf"), "pdf", &sort).0, "Финансы"); } #[test] fn installers_go_to_downloaders() { let sort = SortSettings { mode:"standard".into(), custom_prompt:"".into(), text_limit:1, total_limit:1 }; assert_eq!(classify(Path::new("Discord.dmg"), "dmg", &sort).0, "Загрузчики"); assert_eq!(classify(Path::new("coconut_latest.zip"), "zip", &sort).0, "Загрузчики"); } }
