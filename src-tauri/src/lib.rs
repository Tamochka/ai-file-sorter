use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs,
    future::Future,
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tauri::Emitter;
use uuid::Uuid;
use walkdir::WalkDir;

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct AiSettings {
    provider: String,
    base_url: String,
    model: String,
    api_key: String,
    cloud_consent: bool,
}
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct SortSettings {
    mode: String,
    custom_prompt: String,
    text_limit: usize,
    total_limit: usize,
    #[serde(default)]
    unlimited: bool,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct PlanItem {
    id: String,
    source: String,
    relative_path: String,
    target: String,
    category: String,
    explanation: String,
    confidence: f32,
    included: bool,
    warning: Option<String>,
    ai_status: AiStatus,
    ai_error: Option<String>,
}
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum AiStatus {
    Processed,
    RetryPending,
    Unprocessed,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AnalysisResult {
    items: Vec<PlanItem>,
    total_files: usize,
    estimated_chars: usize,
    warnings: Vec<String>,
    summary: AiSummary,
}
#[derive(Debug, Serialize, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct AiSummary {
    ai_processed: usize,
    retry_succeeded: usize,
    ai_unprocessed: usize,
}
#[derive(Debug, Serialize, Deserialize)]
struct MoveRecord {
    from: String,
    to: String,
}
#[derive(Debug, Deserialize)]
struct AiDecision {
    id: String,
    category: String,
    explanation: Option<String>,
    confidence: Option<f32>,
}
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct AiFileContext {
    id: String,
    path: String,
    extension: String,
    size_bytes: u64,
    created_at: Option<String>,
    modified_at: Option<String>,
    suggested_category: String,
    content_extract: Option<String>,
    content_status: String,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelList {
    models: Vec<String>,
    active_model: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct AnalysisProgress {
    phase: String,
    completed_batches: usize,
    total_batches: usize,
    processed_files: usize,
    pending_files: usize,
    message: String,
}

#[derive(Default)]
struct AnalysisControl {
    cancelled: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
}

struct AnalysisGuard(Arc<AtomicBool>);

impl Drop for AnalysisGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

struct PreparedAnalysis {
    items: Vec<PlanItem>,
    contexts: Vec<AiFileContext>,
    total_chars: usize,
    warnings: Vec<String>,
}

const SORTED_DIR: &str = "AI Sorted";
const UNPROCESSED_CATEGORY: &str = "Не обработано ИИ";
const HISTORY_FILE: &str = ".ai-file-sorter-last-operation.json";
const AI_BATCH_SIZE: usize = 10;
const AI_BATCH_TIMEOUT: Duration = Duration::from_secs(90);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

#[tauri::command]
async fn analyze_folder(
    app: tauri::AppHandle,
    control: tauri::State<'_, AnalysisControl>,
    folder: String,
    ai: AiSettings,
    sort: SortSettings,
) -> Result<AnalysisResult, String> {
    if ai.model.trim().is_empty() {
        return Err("Укажите имя модели".into());
    }
    let is_local = ai.provider == "lmstudio"
        || ai.provider == "ollama"
        || ai.base_url.contains("localhost")
        || ai.base_url.contains("127.0.0.1");
    if !is_local && !ai.cloud_consent {
        return Err("Нужно подтвердить передачу данных в облако".into());
    }
    if control
        .running
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err("Анализ уже выполняется".into());
    }
    control.cancelled.store(false, Ordering::Release);
    let _guard = AnalysisGuard(control.running.clone());
    let cancelled = control.cancelled.clone();

    emit_progress(
        &app,
        AnalysisProgress {
            phase: "scanning".into(),
            completed_batches: 0,
            total_batches: 0,
            processed_files: 0,
            pending_files: 0,
            message: "Сканирование файлов…".into(),
        },
    );

    let scan_cancelled = cancelled.clone();
    let scan_sort = sort.clone();
    let prepared = tauri::async_runtime::spawn_blocking(move || {
        prepare_analysis(folder, &scan_sort, &scan_cancelled)
    })
    .await
    .map_err(|error| format!("Фоновый анализ завершился аварийно: {error}"))??;

    if cancelled.load(Ordering::Acquire) {
        return Err("Анализ отменён пользователем".into());
    }

    let client = reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(AI_BATCH_TIMEOUT + Duration::from_secs(5))
        .build()
        .map_err(|error| format!("Не удалось создать HTTP-клиент: {error}"))?;

    let PreparedAnalysis {
        mut items,
        contexts,
        total_chars,
        mut warnings,
    } = prepared;
    let progress_app = app.clone();
    let refinement = refine_with_model(
        &client,
        &ai,
        &sort,
        &mut items,
        &contexts,
        cancelled,
        move |progress| emit_progress(&progress_app, progress),
    )
    .await?;
    warnings.extend(refinement.warnings);
    if !sort.unlimited && total_chars >= sort.total_limit {
        warnings.push(
            "Достигнут общий лимит текста. Часть файлов будет оценена по имени и метаданным."
                .into(),
        );
    }
    Ok(AnalysisResult {
        total_files: items.len(),
        estimated_chars: total_chars,
        items,
        warnings,
        summary: refinement.summary,
    })
}

fn prepare_analysis(
    folder: String,
    sort: &SortSettings,
    cancelled: &AtomicBool,
) -> Result<PreparedAnalysis, String> {
    let root = canonical_root(&folder)?;
    let mut total_chars = 0usize;
    let mut items = Vec::new();
    let mut contexts = Vec::new();
    let mut warnings = Vec::new();
    for entry in WalkDir::new(&root)
        .into_iter()
        .filter_entry(scan_entry)
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        if cancelled.load(Ordering::Acquire) {
            return Err("Анализ отменён пользователем".into());
        }
        let path = entry.path();
        let relative = path
            .strip_prefix(&root)
            .map_err(|_| "Не удалось вычислить относительный путь")?;
        if skip_file(path) {
            continue;
        }
        let metadata = match fs::metadata(path) {
            Ok(data) => data,
            Err(_) => {
                warnings.push(format!("Нет доступа к {}", relative.display()));
                continue;
            }
        };
        let ext = path
            .extension()
            .and_then(|x| x.to_str())
            .unwrap_or("")
            .to_lowercase();
        let remaining = if sort.unlimited {
            usize::MAX
        } else {
            sort.total_limit.saturating_sub(total_chars)
        };
        let per_file_limit = if sort.unlimited {
            usize::MAX
        } else {
            sort.text_limit
        };
        let (content_extract, content_status) =
            read_text_preview(path, &ext, per_file_limit.min(remaining));
        total_chars = total_chars.saturating_add(
            content_extract
                .as_ref()
                .map_or(0, |text| text.chars().count()),
        );
        let (category, confidence, explanation) = classify(relative, &ext, sort);
        let date = metadata
            .modified()
            .ok()
            .map(|t| DateTime::<Local>::from(t).format("%Y/%m").to_string())
            .unwrap_or_else(|| "Без даты".into());
        let kind = file_kind(&ext);
        let target = Path::new(SORTED_DIR)
            .join(safe_name(&category))
            .join(kind)
            .join(date)
            .join(relative);
        let id = Uuid::new_v4().to_string();
        contexts.push(AiFileContext {
            id: id.clone(),
            path: relative.to_string_lossy().into_owned(),
            extension: ext.clone(),
            size_bytes: metadata.len(),
            created_at: format_file_time(metadata.created().ok()),
            modified_at: format_file_time(metadata.modified().ok()),
            suggested_category: category.clone(),
            content_extract,
            content_status,
        });
        items.push(PlanItem {
            id,
            source: path.to_string_lossy().into_owned(),
            relative_path: relative.to_string_lossy().into_owned(),
            target: target.to_string_lossy().into_owned(),
            category,
            explanation,
            confidence,
            included: true,
            warning: unsupported_warning(&ext),
            ai_status: AiStatus::RetryPending,
            ai_error: None,
        });
    }
    if sort.mode == "custom" && sort.custom_prompt.trim().is_empty() {
        warnings.push("Кастомный режим без инструкции использовал стандартные категории.".into());
    }
    Ok(PreparedAnalysis {
        items,
        contexts,
        total_chars,
        warnings,
    })
}

#[tauri::command]
fn cancel_analysis(control: tauri::State<'_, AnalysisControl>) -> bool {
    let running = control.running.load(Ordering::Acquire);
    if running {
        control.cancelled.store(true, Ordering::Release);
    }
    running
}

fn emit_progress(app: &tauri::AppHandle, progress: AnalysisProgress) {
    let _ = app.emit("analysis-progress", progress);
}

#[tauri::command]
fn apply_sort(folder: String, items: Vec<PlanItem>) -> Result<usize, String> {
    let root = canonical_root(&folder)?;
    let mut records = Vec::new();
    let mut seen = HashSet::new();
    for item in items {
        let source = canonical_inside(&root, Path::new(&item.source))?;
        let destination = safe_destination(&root, &item.target)?;
        if source == destination {
            continue;
        }
        if !seen.insert(destination.clone()) {
            return Err(format!(
                "Повторяющийся целевой путь: {}",
                destination.display()
            ));
        }
        let destination = conflict_free(&destination);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(io_error)?;
        }
        fs::rename(&source, &destination).map_err(io_error)?;
        records.push(MoveRecord {
            from: source.to_string_lossy().into_owned(),
            to: destination.to_string_lossy().into_owned(),
        });
    }
    let history = root.join(HISTORY_FILE);
    fs::write(
        history,
        serde_json::to_vec_pretty(&records).map_err(|e| e.to_string())?,
    )
    .map_err(io_error)?;
    Ok(records.len())
}

#[tauri::command]
fn undo_last_sort(folder: String) -> Result<usize, String> {
    let root = canonical_root(&folder)?;
    let history = root.join(HISTORY_FILE);
    let raw = fs::read(&history).map_err(|_| "Нет операции для отмены".to_string())?;
    let records: Vec<MoveRecord> =
        serde_json::from_slice(&raw).map_err(|_| "Журнал операции повреждён".to_string())?;
    let mut restored = 0;
    for record in records.into_iter().rev() {
        let current = canonical_inside(&root, Path::new(&record.to))?;
        let original = safe_recorded_destination(&root, Path::new(&record.from))?;
        let original = conflict_free(&original);
        if let Some(parent) = original.parent() {
            fs::create_dir_all(parent).map_err(io_error)?;
        }
        fs::rename(current, original).map_err(io_error)?;
        restored += 1;
    }
    fs::remove_file(history).map_err(io_error)?;
    Ok(restored)
}

#[tauri::command]
async fn test_connection(ai: AiSettings) -> Result<String, String> {
    if ai.base_url.trim().is_empty() {
        return Err("Укажите базовый URL".into());
    }
    let url = format!("{}/models", ai.base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(8))
        .build()
        .map_err(|error| format!("Не удалось создать HTTP-клиент: {error}"))?;
    let mut request = client.get(&url);
    if !ai.api_key.trim().is_empty() {
        request = request.bearer_auth(&ai.api_key);
    }
    let response = tokio::time::timeout(Duration::from_secs(8), request.send())
        .await
        .map_err(|_| "Сервис не ответил за 8 секунд".to_string())?
        .map_err(|error| format!("Сервис не ответил: {error}"))?;
    response
        .error_for_status()
        .map_err(|error| format!("Сервис вернул ошибку: {error}"))?;
    Ok(format!("Подключение успешно: {}", ai.base_url))
}

#[tauri::command]
async fn list_models(ai: AiSettings) -> Result<ModelList, String> {
    if ai.base_url.trim().is_empty() {
        return Err("Укажите базовый URL".into());
    }
    let url = if ai.provider == "lmstudio" {
        format!(
            "{}/api/v1/models",
            ai.base_url.trim_end_matches('/').trim_end_matches("/v1")
        )
    } else if ai.provider == "ollama" {
        format!(
            "{}/api/tags",
            ai.base_url.trim_end_matches('/').trim_end_matches("/v1")
        )
    } else {
        format!("{}/models", ai.base_url.trim_end_matches('/'))
    };
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(12))
        .build()
        .map_err(|error| format!("Не удалось создать HTTP-клиент: {error}"))?;
    let mut request = client.get(&url);
    if !ai.api_key.trim().is_empty() {
        request = request.bearer_auth(&ai.api_key);
    }
    let response = tokio::time::timeout(Duration::from_secs(12), request.send())
        .await
        .map_err(|_| "Список моделей не получен за 12 секунд".to_string())?
        .map_err(|error| format!("Не удалось получить модели: {error}"))?
        .error_for_status()
        .map_err(|error| format!("Сервис вернул ошибку: {error}"))?;
    let value: serde_json::Value = response
        .json()
        .await
        .map_err(|error| format!("Список моделей не является JSON: {error}"))?;
    let source = if ai.provider == "lmstudio" || ai.provider == "ollama" {
        value.get("models").and_then(|v| v.as_array())
    } else {
        value.get("data").and_then(|v| v.as_array())
    }
    .ok_or("Сервис вернул список моделей в неизвестном формате")?;
    if ai.provider == "lmstudio" {
        let active_model = source
            .iter()
            .filter(|model| model.get("type").and_then(|kind| kind.as_str()) == Some("llm"))
            .find_map(|model| {
                model
                    .get("loaded_instances")
                    .and_then(|instances| instances.as_array())
                    .and_then(|instances| instances.first())
                    .and_then(|instance| instance.get("id"))
                    .and_then(|id| id.as_str())
                    .map(str::to_owned)
            });
        let mut models: Vec<String> = source
            .iter()
            .filter(|model| model.get("type").and_then(|kind| kind.as_str()) == Some("llm"))
            .filter_map(|model| {
                model
                    .get("key")
                    .and_then(|key| key.as_str())
                    .map(str::to_owned)
            })
            .collect();
        if let Some(active) = &active_model {
            models.retain(|model| model != active);
            models.insert(0, active.clone());
        }
        if models.is_empty() {
            return Err("LM Studio не сообщил доступных языковых моделей".into());
        }
        return Ok(ModelList {
            models,
            active_model,
        });
    }
    let mut models: Vec<String> = source
        .iter()
        .filter_map(|model| {
            model
                .get(if ai.provider == "ollama" {
                    "name"
                } else {
                    "id"
                })
                .and_then(|name| name.as_str())
                .map(str::to_owned)
        })
        .collect();
    models.sort();
    models.dedup();
    if models.is_empty() {
        return Err("Локальный сервис не сообщил доступных моделей".into());
    }
    Ok(ModelList {
        models,
        active_model: None,
    })
}

fn scan_entry(entry: &walkdir::DirEntry) -> bool {
    if entry.depth() == 0 {
        return true;
    }
    let name = entry.file_name().to_string_lossy();
    !(entry.file_type().is_dir()
        && (name == SORTED_DIR || name.to_ascii_lowercase().ends_with(".app")))
}
fn skip_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(".DS_Store") | Some(".localized")
    )
}
fn canonical_root(folder: &str) -> Result<PathBuf, String> {
    let path = fs::canonicalize(folder).map_err(io_error)?;
    if !path.is_dir() {
        return Err("Выбранный путь не является папкой".into());
    }
    Ok(path)
}
fn canonical_inside(root: &Path, candidate: &Path) -> Result<PathBuf, String> {
    let path = fs::canonicalize(candidate).map_err(io_error)?;
    if !path.starts_with(root) {
        return Err("Путь выходит за пределы выбранной папки".into());
    }
    Ok(path)
}
fn safe_destination(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let p = Path::new(relative);
    if p.is_absolute()
        || p.components().any(|c| {
            matches!(
                c,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err("Недопустимый целевой путь".into());
    }
    if p.components().any(|c| {
        c.as_os_str()
            .to_string_lossy()
            .chars()
            .any(|ch| matches!(ch, ':' | '*' | '?' | '"' | '<' | '>' | '|'))
    }) {
        return Err("Недопустимые символы в имени папки или файла".into());
    }
    let out = root.join(p);
    if !out.starts_with(root) {
        return Err("Путь выходит за пределы выбранной папки".into());
    }
    Ok(out)
}
fn safe_recorded_destination(root: &Path, candidate: &Path) -> Result<PathBuf, String> {
    if !candidate.is_absolute() {
        return safe_destination(root, &candidate.to_string_lossy());
    }
    if candidate
        .components()
        .any(|component| matches!(component, Component::ParentDir | Component::Prefix(_)))
        || !candidate.starts_with(root)
    {
        return Err("Путь выходит за пределы выбранной папки".into());
    }
    Ok(candidate.to_path_buf())
}
fn conflict_free(path: &Path) -> PathBuf {
    if !path.exists() {
        return path.to_path_buf();
    }
    let stem = path.file_stem().and_then(|x| x.to_str()).unwrap_or("file");
    let ext = path
        .extension()
        .and_then(|x| x.to_str())
        .map(|x| format!(".{x}"))
        .unwrap_or_default();
    for i in 2.. {
        let candidate = path.with_file_name(format!("{stem} ({i}){ext}"));
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!()
}
fn io_error(e: std::io::Error) -> String {
    e.to_string()
}
fn safe_name(value: &str) -> String {
    let clean: String = value
        .chars()
        .map(|c| {
            if matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') {
                '_'
            } else {
                c
            }
        })
        .collect();
    if clean.trim().is_empty() {
        "Прочее".into()
    } else {
        clean
    }
}
fn file_kind(ext: &str) -> &'static str {
    match ext {
        "dmg" | "pkg" | "mpkg" | "exe" | "msi" | "appimage" | "deb" | "rpm" | "apk" | "iso" => {
            "Установщики"
        }
        "pdf" => "PDF",
        "doc" | "docx" | "odt" => "Документы",
        "xls" | "xlsx" | "csv" => "Таблицы",
        "txt" | "md" | "rtf" => "Текст",
        "jpg" | "jpeg" | "png" | "webp" | "heic" => "Изображения",
        "mp3" | "wav" | "m4a" => "Аудио",
        "mp4" | "mov" | "mkv" => "Видео",
        _ => "Прочие файлы",
    }
}
fn unsupported_warning(ext: &str) -> Option<String> {
    if matches!(ext, "mp3" | "wav" | "m4a" | "mp4" | "mov" | "mkv") {
        Some("Содержимое аудио/видео в первой версии не анализируется.".into())
    } else {
        None
    }
}
fn format_file_time(time: Option<std::time::SystemTime>) -> Option<String> {
    time.map(|t| DateTime::<Local>::from(t).to_rfc3339())
}
fn read_text_preview(path: &Path, ext: &str, limit: usize) -> (Option<String>, String) {
    if !matches!(
        ext,
        "txt"
            | "md"
            | "csv"
            | "json"
            | "xml"
            | "yaml"
            | "yml"
            | "toml"
            | "html"
            | "htm"
            | "log"
            | "py"
            | "js"
            | "ts"
            | "rs"
            | "java"
            | "c"
            | "cpp"
            | "h"
            | "sql"
    ) {
        return (
            None,
            "Содержимое этого формата не извлекается; использованы метаданные файла.".into(),
        );
    }
    if limit == 0 {
        return (
            None,
            "Достигнут общий лимит текста; использованы метаданные файла.".into(),
        );
    }
    match fs::read_to_string(path) {
        Ok(text) => {
            let excerpt: String = text.chars().take(limit).collect();
            if excerpt.is_empty() {
                (
                    None,
                    "Текстовый файл пуст; использованы метаданные файла.".into(),
                )
            } else {
                (
                    Some(excerpt),
                    "Передан фрагмент текстового содержимого и метаданные файла.".into(),
                )
            }
        }
        Err(_) => (
            None,
            "Не удалось прочитать содержимое; использованы метаданные файла.".into(),
        ),
    }
}
fn classify(relative: &Path, ext: &str, sort: &SortSettings) -> (String, f32, String) {
    let name = relative.to_string_lossy().to_lowercase();
    let installer = matches!(
        ext,
        "dmg" | "pkg" | "mpkg" | "exe" | "msi" | "appimage" | "deb" | "rpm" | "apk" | "iso"
    ) || (matches!(ext, "zip" | "7z")
        && [
            "setup",
            "install",
            "installer",
            "latest",
            "download",
            "tsetup",
        ]
        .iter()
        .any(|word| name.contains(word)));
    let category = if sort.mode == "custom" && !sort.custom_prompt.trim().is_empty() {
        "Кастомная категория"
    } else if installer {
        "Загрузчики"
    } else if name.contains("invoice")
        || name.contains("счёт")
        || name.contains("налог")
        || name.contains("bank")
    {
        "Финансы"
    } else if name.contains("resume")
        || name.contains("проект")
        || name.contains("contract")
        || name.contains("договор")
    {
        "Работа"
    } else if name.contains("course")
        || name.contains("лекц")
        || name.contains("study")
        || name.contains("учеб")
    {
        "Учёба"
    } else if matches!(
        ext,
        "jpg" | "jpeg" | "png" | "webp" | "mp4" | "mov" | "mkv" | "mp3"
    ) {
        "Медиа"
    } else if matches!(ext, "zip" | "rar" | "7z" | "tar" | "gz") {
        "Архив"
    } else {
        "Личное"
    };
    let confidence = if category == "Личное" {
        0.45
    } else if category == "Загрузчики" {
        0.95
    } else {
        0.72
    };
    let explanation = if category == "Загрузчики" {
        "Распознан установочный или загрузочный файл по расширению либо имени."
    } else {
        "Предварительная локальная оценка по имени, расширению и дате."
    };
    (category.into(), confidence, explanation.into())
}

#[derive(Debug, Default)]
struct AiRefinement {
    summary: AiSummary,
    warnings: Vec<String>,
}

#[derive(Debug)]
enum BatchError {
    Failure(String),
    Cancelled,
}

async fn refine_with_model<P>(
    client: &reqwest::Client,
    ai: &AiSettings,
    sort: &SortSettings,
    items: &mut [PlanItem],
    contexts: &[AiFileContext],
    cancelled: Arc<AtomicBool>,
    progress: P,
) -> Result<AiRefinement, String>
where
    P: FnMut(AnalysisProgress),
{
    let request_cancelled = cancelled.clone();
    refine_in_batches(
        sort,
        items,
        contexts,
        move |batch| {
            request_model_batch(
                client,
                ai,
                sort,
                batch,
                request_cancelled.clone(),
                AI_BATCH_TIMEOUT,
            )
        },
        progress,
    )
    .await
}

async fn request_model_batch(
    client: &reqwest::Client,
    ai: &AiSettings,
    sort: &SortSettings,
    batch: Vec<AiFileContext>,
    cancelled: Arc<AtomicBool>,
    hard_timeout: Duration,
) -> Result<Vec<AiDecision>, BatchError> {
    if cancelled.load(Ordering::Acquire) {
        return Err(BatchError::Cancelled);
    }
    let instruction = if sort.mode == "custom" {
        sort.custom_prompt.as_str()
    } else {
        "Используй только категории: Работа, Личное, Финансы, Учёба, Медиа, Архив, Загрузчики, Прочее. Установочные файлы с расширениями DMG, EXE, PKG, MSI и похожими всегда относятся к Загрузчикам."
    };
    let url = format!("{}/chat/completions", ai.base_url.trim_end_matches('/'));
    let prompt = format!("Классифицируй только этот небольшой пакет файлов. Инструкция: {instruction}\nДля каждого файла сначала используй contentExtract, если он есть. Если его нет, анализируй только метаданные: path, extension, sizeBytes, даты и suggestedCategory. Не выдумывай содержимое. Верни ТОЛЬКО JSON-массив объектов {{id, category, explanation, confidence}}. Верни ровно одно решение для каждого переданного id. category — короткое безопасное имя папки без / и \\.\nФайлы: {}", serde_json::to_string(&batch).map_err(|error| BatchError::Failure(error.to_string()))?);
    let body = serde_json::json!({"model":ai.model,"temperature":0,"messages":[{"role":"system","content":"Ты отвечаешь строго валидным JSON без Markdown."},{"role":"user","content":prompt}]});
    let mut request = client.post(&url).json(&body);
    if !ai.api_key.trim().is_empty() {
        request = request.bearer_auth(&ai.api_key);
    }
    let operation = async {
        let response = request
            .send()
            .await
            .map_err(|error| BatchError::Failure(format!("Ошибка запроса к модели: {error}")))?
            .error_for_status()
            .map_err(|error| BatchError::Failure(format!("Модель вернула ошибку HTTP: {error}")))?;
        let value: serde_json::Value = response
            .json()
            .await
            .map_err(|error| BatchError::Failure(format!("Ответ API не является JSON: {error}")))?;
        parse_model_response(&value).map_err(BatchError::Failure)
    };

    tokio::select! {
        _ = wait_for_cancellation(cancelled) => Err(BatchError::Cancelled),
        outcome = tokio::time::timeout(hard_timeout, operation) => {
            outcome.map_err(|_| BatchError::Failure(format!(
                "Тайм-аут пакета: модель не ответила за {} секунд",
                hard_timeout.as_secs()
            )))?
        }
    }
}

async fn wait_for_cancellation(cancelled: Arc<AtomicBool>) {
    while !cancelled.load(Ordering::Acquire) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn parse_model_response(value: &serde_json::Value) -> Result<Vec<AiDecision>, String> {
    let content = value
        .pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .ok_or("Ответ API не содержит choices[0].message.content")?;
    let trimmed = content.trim();
    let without_opening = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed);
    let json = without_opening
        .trim()
        .strip_suffix("```")
        .unwrap_or(without_opening.trim())
        .trim();
    serde_json::from_str(json).map_err(|error| format!("Модель вернула невалидный JSON: {error}"))
}

async fn refine_in_batches<F, Fut, P>(
    sort: &SortSettings,
    items: &mut [PlanItem],
    contexts: &[AiFileContext],
    mut request_batch: F,
    mut progress: P,
) -> Result<AiRefinement, String>
where
    F: FnMut(Vec<AiFileContext>) -> Fut,
    Fut: Future<Output = Result<Vec<AiDecision>, BatchError>>,
    P: FnMut(AnalysisProgress),
{
    if items.is_empty() {
        progress(AnalysisProgress {
            phase: "complete".into(),
            completed_batches: 0,
            total_batches: 0,
            processed_files: 0,
            pending_files: 0,
            message: "В выбранной папке нет файлов для анализа.".into(),
        });
        return Ok(AiRefinement::default());
    }
    let mut result = AiRefinement::default();
    let mut first_failures = HashMap::<String, String>::new();
    let main_batches = contexts.len().div_ceil(AI_BATCH_SIZE);

    for (index, batch) in contexts.chunks(AI_BATCH_SIZE).enumerate() {
        progress(AnalysisProgress {
            phase: "main".into(),
            completed_batches: index,
            total_batches: main_batches,
            processed_files: processed_count(items),
            pending_files: retry_pending_count(items),
            message: format!("Основной проход: пакет {} из {main_batches}…", index + 1),
        });
        match request_batch(batch.to_vec()).await {
            Ok(decisions) => {
                let missing = apply_batch_decisions(sort, items, batch, decisions);
                if !missing.is_empty() {
                    let reason = "Модель вернула частичный ответ: решение для файла отсутствует."
                        .to_string();
                    for id in &missing {
                        first_failures.insert(id.clone(), reason.clone());
                    }
                    result.warnings.push(format!("Основной проход, пакет {}: модель не вернула решения для {} файлов; они отправлены на повторную попытку.", index + 1, missing.len()));
                }
            }
            Err(BatchError::Failure(error)) => {
                for context in batch {
                    first_failures.insert(context.id.clone(), error.clone());
                }
                result.warnings.push(format!("Основной проход, пакет {}: {error}. На повторную попытку отправлено {} файлов.", index + 1, batch.len()));
            }
            Err(BatchError::Cancelled) => return Err("Анализ отменён пользователем".into()),
        }
        progress(AnalysisProgress {
            phase: "main".into(),
            completed_batches: index + 1,
            total_batches: main_batches,
            processed_files: processed_count(items),
            pending_files: retry_pending_count(items),
            message: format!(
                "Основной проход: завершено пакетов {} из {main_batches}.",
                index + 1
            ),
        });
    }

    let retry_contexts: Vec<AiFileContext> = contexts
        .iter()
        .filter(|context| item_status(items, &context.id) == Some(AiStatus::RetryPending))
        .cloned()
        .collect();
    let retry_batches = retry_contexts.len().div_ceil(AI_BATCH_SIZE);
    for (index, batch) in retry_contexts.chunks(AI_BATCH_SIZE).enumerate() {
        progress(AnalysisProgress {
            phase: "retry".into(),
            completed_batches: index,
            total_batches: retry_batches,
            processed_files: processed_count(items),
            pending_files: retry_pending_count(items),
            message: format!("Повторный проход: пакет {} из {retry_batches}…", index + 1),
        });
        match request_batch(batch.to_vec()).await {
            Ok(decisions) => {
                let missing = apply_batch_decisions(sort, items, batch, decisions);
                result.summary.retry_succeeded += batch.len() - missing.len();
                if !missing.is_empty() {
                    let reason =
                        "Модель снова вернула частичный ответ: решение для файла отсутствует.";
                    for id in &missing {
                        mark_unprocessed(items, id, first_failures.get(id), reason);
                    }
                    result.warnings.push(format!("Повторный проход, пакет {}: для {} файлов снова нет решения; они направлены в «{}».", index + 1, missing.len(), UNPROCESSED_CATEGORY));
                }
            }
            Err(BatchError::Failure(error)) => {
                for context in batch {
                    mark_unprocessed(items, &context.id, first_failures.get(&context.id), &error);
                }
                result.warnings.push(format!(
                    "Повторный проход, пакет {}: {error}. В «{}» направлено {} файлов.",
                    index + 1,
                    UNPROCESSED_CATEGORY,
                    batch.len()
                ));
            }
            Err(BatchError::Cancelled) => return Err("Анализ отменён пользователем".into()),
        }
        progress(AnalysisProgress {
            phase: "retry".into(),
            completed_batches: index + 1,
            total_batches: retry_batches,
            processed_files: processed_count(items),
            pending_files: retry_pending_count(items),
            message: format!(
                "Повторный проход: завершено пакетов {} из {retry_batches}.",
                index + 1
            ),
        });
    }

    result.summary.ai_processed = items
        .iter()
        .filter(|item| item.ai_status == AiStatus::Processed)
        .count();
    result.summary.ai_unprocessed = items
        .iter()
        .filter(|item| item.ai_status == AiStatus::Unprocessed)
        .count();
    progress(AnalysisProgress {
        phase: "complete".into(),
        completed_batches: main_batches + retry_batches,
        total_batches: main_batches + retry_batches,
        processed_files: result.summary.ai_processed,
        pending_files: 0,
        message: format!(
            "Анализ завершён: ИИ обработал {}, после повтора — {}, не обработано — {}.",
            result.summary.ai_processed,
            result.summary.retry_succeeded,
            result.summary.ai_unprocessed
        ),
    });
    Ok(result)
}

fn processed_count(items: &[PlanItem]) -> usize {
    items
        .iter()
        .filter(|item| item.ai_status == AiStatus::Processed)
        .count()
}

fn retry_pending_count(items: &[PlanItem]) -> usize {
    items
        .iter()
        .filter(|item| item.ai_status == AiStatus::RetryPending)
        .count()
}

fn item_status(items: &[PlanItem], id: &str) -> Option<AiStatus> {
    items
        .iter()
        .find(|item| item.id == id)
        .map(|item| item.ai_status)
}

fn apply_batch_decisions(
    sort: &SortSettings,
    items: &mut [PlanItem],
    batch: &[AiFileContext],
    decisions: Vec<AiDecision>,
) -> Vec<String> {
    let batch_ids: HashSet<&str> = batch.iter().map(|context| context.id.as_str()).collect();
    let mut decisions_by_id = HashMap::new();
    for decision in decisions {
        if batch_ids.contains(decision.id.as_str()) {
            decisions_by_id
                .entry(decision.id.clone())
                .or_insert(decision);
        }
    }
    let mut missing = Vec::new();
    for context in batch {
        if let Some(decision) = decisions_by_id.remove(&context.id) {
            apply_ai_decision(sort, items, decision);
        } else {
            missing.push(context.id.clone());
        }
    }
    missing
}

fn apply_ai_decision(sort: &SortSettings, items: &mut [PlanItem], decision: AiDecision) {
    let Some(item) = items.iter_mut().find(|item| item.id == decision.id) else {
        return;
    };
    if !(sort.mode == "standard" && item.category == "Загрузчики") {
        let category = safe_name(decision.category.trim());
        item.category = category.clone();
        retarget_item(item, &category);
    }
    item.explanation = decision
        .explanation
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "Категория назначена ИИ.".into());
    if let Some(confidence) = decision.confidence {
        item.confidence = confidence.clamp(0.0, 1.0);
    }
    item.ai_status = AiStatus::Processed;
    item.ai_error = None;
}

fn mark_unprocessed(
    items: &mut [PlanItem],
    id: &str,
    first_error: Option<&String>,
    retry_error: &str,
) {
    let Some(item) = items.iter_mut().find(|item| item.id == id) else {
        return;
    };
    let first_error = first_error
        .map(String::as_str)
        .unwrap_or("модель не вернула решение");
    let detail = format!("ИИ не обработал файл после двух попыток. Первая попытка: {first_error} Повторная попытка: {retry_error}");
    item.category = UNPROCESSED_CATEGORY.into();
    retarget_item(item, UNPROCESSED_CATEGORY);
    item.explanation = detail.clone();
    item.confidence = 0.0;
    item.included = true;
    item.ai_status = AiStatus::Unprocessed;
    item.ai_error = Some(detail);
}

fn retarget_item(item: &mut PlanItem, category: &str) {
    let previous = Path::new(&item.target);
    let mut revised = PathBuf::from(SORTED_DIR).join(category);
    for component in previous.components().skip(2) {
        revised.push(component.as_os_str());
    }
    item.target = revised.to_string_lossy().into_owned();
}

pub fn run() {
    tauri::Builder::default()
        .manage(AnalysisControl::default())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            analyze_folder,
            cancel_analysis,
            apply_sort,
            undo_last_sort,
            test_connection,
            list_models
        ])
        .run(tauri::generate_context!())
        .expect("ошибка запуска Tauri");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sort() -> SortSettings {
        SortSettings {
            mode: "standard".into(),
            custom_prompt: "".into(),
            text_limit: 1,
            total_limit: 1,
            unlimited: false,
        }
    }

    fn plan(count: usize) -> (Vec<PlanItem>, Vec<AiFileContext>) {
        let mut items = Vec::new();
        let mut contexts = Vec::new();
        for index in 0..count {
            let id = format!("file-{index}");
            let relative = format!("исходная/папка/file-{index}.txt");
            items.push(PlanItem {
                id: id.clone(),
                source: format!("/tmp/root/{relative}"),
                relative_path: relative.clone(),
                target: format!("{SORTED_DIR}/Личное/Текст/2026/08/{relative}"),
                category: "Личное".into(),
                explanation: "Локальная оценка".into(),
                confidence: 0.45,
                included: true,
                warning: None,
                ai_status: AiStatus::RetryPending,
                ai_error: None,
            });
            contexts.push(AiFileContext {
                id,
                path: relative,
                extension: "txt".into(),
                size_bytes: 10,
                created_at: None,
                modified_at: None,
                suggested_category: "Личное".into(),
                content_extract: Some("текст".into()),
                content_status: "прочитан".into(),
            });
        }
        (items, contexts)
    }

    fn decisions(batch: &[AiFileContext]) -> Vec<AiDecision> {
        batch
            .iter()
            .map(|context| AiDecision {
                id: context.id.clone(),
                category: "Работа".into(),
                explanation: Some("Решение модели".into()),
                confidence: Some(0.9),
            })
            .collect()
    }

    #[tokio::test]
    async fn all_batches_are_processed_successfully() {
        let (mut items, contexts) = plan(23);
        let mut calls = 0;
        let result = refine_in_batches(
            &test_sort(),
            &mut items,
            &contexts,
            |batch| {
                calls += 1;
                std::future::ready(Ok(decisions(&batch)))
            },
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(calls, 3);
        assert_eq!(
            result.summary,
            AiSummary {
                ai_processed: 23,
                retry_succeeded: 0,
                ai_unprocessed: 0
            }
        );
        assert!(items
            .iter()
            .all(|item| item.ai_status == AiStatus::Processed));
    }

    #[tokio::test]
    async fn failed_batch_does_not_stop_following_batches() {
        let (mut items, contexts) = plan(23);
        let mut calls = 0;
        let result = refine_in_batches(
            &test_sort(),
            &mut items,
            &contexts,
            |batch| {
                calls += 1;
                let response = if calls == 2 {
                    Err(BatchError::Failure("Тайм-аут основного пакета".into()))
                } else {
                    Ok(decisions(&batch))
                };
                std::future::ready(response)
            },
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(calls, 4);
        assert_eq!(
            result.summary,
            AiSummary {
                ai_processed: 23,
                retry_succeeded: 10,
                ai_unprocessed: 0
            }
        );
        assert_eq!(items[20].explanation, "Решение модели");
        assert_eq!(items[10].ai_status, AiStatus::Processed);
    }

    #[tokio::test]
    async fn second_failure_moves_only_failed_files_to_unprocessed() {
        let (mut items, contexts) = plan(15);
        let mut calls = 0;
        let result = refine_in_batches(
            &test_sort(),
            &mut items,
            &contexts,
            |batch| {
                calls += 1;
                let response = match calls {
                    1 => Ok(decisions(&batch)),
                    2 => Err(BatchError::Failure("Тайм-аут запроса".into())),
                    _ => Err(BatchError::Failure("Модель вернула невалидный JSON".into())),
                };
                std::future::ready(response)
            },
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(
            result.summary,
            AiSummary {
                ai_processed: 10,
                retry_succeeded: 0,
                ai_unprocessed: 5
            }
        );
        assert!(items[..10]
            .iter()
            .all(|item| item.ai_status == AiStatus::Processed));
        for item in &items[10..] {
            assert_eq!(item.ai_status, AiStatus::Unprocessed);
            assert_eq!(item.category, UNPROCESSED_CATEGORY);
            assert!(item
                .target
                .starts_with("AI Sorted/Не обработано ИИ/Текст/2026/08/исходная/папка/"));
            assert!(item.included);
            let error = item.ai_error.as_deref().unwrap();
            assert!(error.contains("Тайм-аут запроса"));
            assert!(error.contains("невалидный JSON"));
        }
    }

    #[tokio::test]
    async fn missing_partial_decisions_are_retried() {
        let (mut items, contexts) = plan(12);
        let mut calls = 0;
        let result = refine_in_batches(
            &test_sort(),
            &mut items,
            &contexts,
            |batch| {
                calls += 1;
                let response = if calls == 1 {
                    Ok(decisions(&batch[..8]))
                } else {
                    Ok(decisions(&batch))
                };
                std::future::ready(response)
            },
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(calls, 3);
        assert_eq!(
            result.summary,
            AiSummary {
                ai_processed: 12,
                retry_succeeded: 2,
                ai_unprocessed: 0
            }
        );
        assert!(result
            .warnings
            .iter()
            .any(|warning| warning.contains("частичный ответ")
                || warning.contains("не вернула решения")));
    }

    #[test]
    fn invalid_json_has_a_clear_technical_error() {
        let value = serde_json::json!({"choices":[{"message":{"content":"not json"}}]});
        assert!(parse_model_response(&value)
            .unwrap_err()
            .contains("невалидный JSON"));
    }

    #[tokio::test]
    async fn hanging_server_is_stopped_by_the_hard_batch_timeout() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (_stream, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(5)).await;
        });
        let client = reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(Duration::from_millis(100))
            .build()
            .unwrap();
        let ai = AiSettings {
            provider: "compatible".into(),
            base_url: format!("http://{address}"),
            model: "test".into(),
            api_key: String::new(),
            cloud_consent: true,
        };
        let (_, contexts) = plan(1);
        let started = std::time::Instant::now();
        let result = request_model_batch(
            &client,
            &ai,
            &test_sort(),
            contexts,
            Arc::new(AtomicBool::new(false)),
            Duration::from_millis(100),
        )
        .await;
        server.abort();
        assert!(started.elapsed() < Duration::from_secs(1));
        match result {
            Err(BatchError::Failure(error)) => assert!(error.contains("Тайм-аут пакета")),
            _ => panic!("ожидалась техническая ошибка жёсткого тайм-аута"),
        }
    }

    #[tokio::test]
    async fn cancellation_stops_a_batch_before_the_request() {
        let client = reqwest::Client::new();
        let ai = AiSettings {
            provider: "compatible".into(),
            base_url: "http://127.0.0.1:9".into(),
            model: "test".into(),
            api_key: String::new(),
            cloud_consent: true,
        };
        let (_, contexts) = plan(1);
        let result = request_model_batch(
            &client,
            &ai,
            &test_sort(),
            contexts,
            Arc::new(AtomicBool::new(true)),
            Duration::from_secs(1),
        )
        .await;
        assert!(matches!(result, Err(BatchError::Cancelled)));
    }

    #[tokio::test]
    async fn cancellation_interrupts_an_in_flight_request() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (_stream, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(5)).await;
        });
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let ai = AiSettings {
            provider: "compatible".into(),
            base_url: format!("http://{address}"),
            model: "test".into(),
            api_key: String::new(),
            cloud_consent: true,
        };
        let (_, contexts) = plan(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancellation_signal = cancelled.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            cancellation_signal.store(true, Ordering::Release);
        });
        let started = std::time::Instant::now();
        let result = request_model_batch(
            &client,
            &ai,
            &test_sort(),
            contexts,
            cancelled,
            Duration::from_secs(5),
        )
        .await;
        server.abort();
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(matches!(result, Err(BatchError::Cancelled)));
    }

    #[test]
    fn target_rejects_escape() {
        assert!(safe_destination(Path::new("/tmp/root"), "../out").is_err());
        assert!(safe_recorded_destination(
            Path::new("/tmp/root"),
            Path::new("/tmp/outside/file.txt")
        )
        .is_err());
    }

    #[test]
    fn apply_and_undo_restores_the_original_file() {
        let root = std::env::temp_dir().join(format!("ai-file-sorter-undo-{}", Uuid::new_v4()));
        let source = root.join("входящие/file.txt");
        fs::create_dir_all(source.parent().unwrap()).unwrap();
        fs::write(&source, "test").unwrap();
        let item = PlanItem {
            id: "undo".into(),
            source: source.to_string_lossy().into_owned(),
            relative_path: "входящие/file.txt".into(),
            target: "AI Sorted/Работа/Текст/2026/08/входящие/file.txt".into(),
            category: "Работа".into(),
            explanation: "Решение модели".into(),
            confidence: 0.9,
            included: true,
            warning: None,
            ai_status: AiStatus::Processed,
            ai_error: None,
        };
        assert_eq!(
            apply_sort(root.to_string_lossy().into_owned(), vec![item]).unwrap(),
            1
        );
        assert!(!source.exists());
        assert_eq!(
            undo_last_sort(root.to_string_lossy().into_owned()).unwrap(),
            1
        );
        assert_eq!(fs::read_to_string(&source).unwrap(), "test");
        assert!(!root.join(HISTORY_FILE).exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn existing_target_gets_a_conflict_suffix() {
        let root = std::env::temp_dir().join(format!("ai-file-sorter-conflict-{}", Uuid::new_v4()));
        let source = root.join("file.txt");
        let target = root.join("AI Sorted/Работа/Текст/2026/08/file.txt");
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(&source, "new").unwrap();
        fs::write(&target, "existing").unwrap();
        let item = PlanItem {
            id: "conflict".into(),
            source: source.to_string_lossy().into_owned(),
            relative_path: "file.txt".into(),
            target: "AI Sorted/Работа/Текст/2026/08/file.txt".into(),
            category: "Работа".into(),
            explanation: "Решение модели".into(),
            confidence: 0.9,
            included: true,
            warning: None,
            ai_status: AiStatus::Processed,
            ai_error: None,
        };
        assert_eq!(
            apply_sort(root.to_string_lossy().into_owned(), vec![item]).unwrap(),
            1
        );
        assert_eq!(
            fs::read_to_string(target.with_file_name("file (2).txt")).unwrap(),
            "new"
        );
        assert_eq!(fs::read_to_string(&target).unwrap(), "existing");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn scan_excludes_sorted_and_app_directories() {
        let root = std::env::temp_dir().join(format!("ai-file-sorter-scan-{}", Uuid::new_v4()));
        fs::create_dir_all(root.join("AI Sorted/old")).unwrap();
        fs::create_dir_all(root.join("Example.app/Contents")).unwrap();
        fs::write(root.join("keep.txt"), "keep").unwrap();
        fs::write(root.join("AI Sorted/old/skip.txt"), "skip").unwrap();
        fs::write(root.join("Example.app/Contents/skip.txt"), "skip").unwrap();
        let files: Vec<PathBuf> = WalkDir::new(&root)
            .into_iter()
            .filter_entry(scan_entry)
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .map(|entry| entry.into_path())
            .collect();
        assert_eq!(files, vec![root.join("keep.txt")]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn names_are_sanitized() {
        assert_eq!(safe_name("A/B: C"), "A_B_ C");
    }

    #[test]
    fn standard_categories_work() {
        assert_eq!(
            classify(Path::new("tax_invoice.pdf"), "pdf", &test_sort()).0,
            "Финансы"
        );
    }

    #[test]
    fn installers_go_to_downloaders() {
        let sort = test_sort();
        assert_eq!(
            classify(Path::new("Discord.dmg"), "dmg", &sort).0,
            "Загрузчики"
        );
        assert_eq!(
            classify(Path::new("coconut_latest.zip"), "zip", &sort).0,
            "Загрузчики"
        );
    }
}
