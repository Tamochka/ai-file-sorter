use chrono::{DateTime, Local};
use serde::{Deserialize, Deserializer, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, OpenOptions},
    future::Future,
    io::Read,
    net::IpAddr,
    ops::Range,
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
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

#[derive(Debug, Clone)]
struct PlannedMove {
    from: PathBuf,
    to: PathBuf,
}
#[derive(Debug, Deserialize)]
struct AiDecision {
    id: String,
    category: String,
    explanation: Option<String>,
    #[serde(default, deserialize_with = "deserialize_confidence")]
    confidence: Option<f32>,
}

fn deserialize_confidence<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    let Some(value) = value else {
        return Ok(None);
    };
    let confidence = match value {
        serde_json::Value::Number(number) => number.as_f64().map(|number| number as f32),
        serde_json::Value::String(value) => {
            let normalized = value.trim().to_lowercase();
            match normalized.as_str() {
                "high" | "высокая" | "высокий" | "высоко" => Some(0.75),
                "medium" | "average" | "средняя" | "средний" | "средне" => {
                    Some(0.5)
                }
                "low" | "низкая" | "низкий" | "низко" => Some(0.25),
                _ => {
                    let is_percent = normalized.ends_with('%');
                    normalized
                        .trim_end_matches('%')
                        .trim()
                        .replace(',', ".")
                        .parse::<f32>()
                        .ok()
                        .map(|number| if is_percent { number / 100.0 } else { number })
                }
            }
        }
        _ => None,
    };
    Ok(confidence)
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
    not_attempted_files: usize,
    retry_pending_files: usize,
    message: String,
}

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct ExtensionCount {
    extension: String,
    count: usize,
}

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct AnalysisLogEvent {
    phase: String,
    attempt: Option<usize>,
    batch_number: Option<usize>,
    total_batches: Option<usize>,
    file_count: usize,
    extensions: Vec<ExtensionCount>,
    duration_ms: u64,
    outcome: String,
    successful_files: usize,
    unresolved_files: usize,
    skipped_files: usize,
    input_bytes: Option<usize>,
    error_kind: Option<String>,
    error_detail: Option<String>,
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
    inaccessible_files: usize,
}

const SORTED_DIR: &str = "AI Sorted";
const UNPROCESSED_CATEGORY: &str = "Не обработано ИИ";
const HISTORY_FILE: &str = ".ai-file-sorter-last-operation.json";
const AI_BATCH_SIZE: usize = 10;
// This cap applies to the whole request, after the user-selected per-file text limit.
// It keeps local models from receiving an unbounded prompt.
const MAX_BATCH_CONTEXT_BYTES: usize = 8_000;
const MAX_MODEL_RESPONSE_TOKENS: usize = 512;
const MAX_CUSTOM_PROMPT_CHARS: usize = 600;
const AI_BATCH_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const STANDARD_CATEGORIES: [&str; 8] = [
    "Работа",
    "Личное",
    "Финансы",
    "Учёба",
    "Медиа",
    "Архив",
    "Загрузчики",
    "Прочее",
];

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
    let is_local = is_loopback_endpoint(&ai.base_url);
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
            not_attempted_files: 0,
            retry_pending_files: 0,
            message: "Сканирование файлов…".into(),
        },
    );

    let scan_started = Instant::now();
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
        inaccessible_files,
    } = prepared;
    emit_analysis_log(
        &app,
        AnalysisLogEvent {
            phase: "scanning".into(),
            attempt: None,
            batch_number: None,
            total_batches: None,
            file_count: contexts.len(),
            extensions: extension_summary(&contexts),
            duration_ms: elapsed_ms(scan_started),
            outcome: "success".into(),
            successful_files: contexts.len(),
            unresolved_files: 0,
            skipped_files: inaccessible_files,
            input_bytes: None,
            error_kind: None,
            error_detail: None,
        },
    );
    let progress_app = app.clone();
    let log_app = app.clone();
    let refinement = refine_with_model(
        &client,
        &ai,
        &sort,
        &mut items,
        &contexts,
        cancelled,
        (
            move |progress| emit_progress(&progress_app, progress),
            move |event| emit_analysis_log(&log_app, event),
        ),
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
    let mut inaccessible_files = 0usize;
    if sort.unlimited {
        warnings.push(format!(
            "Без общего лимита: для каждого файла используется заданный лимит текста, а один запрос к ИИ ограничен {MAX_BATCH_CONTEXT_BYTES} байтами контекста."
        ));
    }
    if sort.mode == "custom" && sort.custom_prompt.chars().count() > MAX_CUSTOM_PROMPT_CHARS {
        warnings.push(format!(
            "Для устойчивости в ИИ передаются только первые {MAX_CUSTOM_PROMPT_CHARS} символов пользовательской инструкции."
        ));
    }
    for entry in WalkDir::new(&root).into_iter().filter_entry(scan_entry) {
        if cancelled.load(Ordering::Acquire) {
            return Err("Анализ отменён пользователем".into());
        }
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => {
                inaccessible_files += 1;
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
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
                inaccessible_files += 1;
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
        let per_file_limit = sort.text_limit;
        let (content_extract, content_status) =
            read_text_preview(path, &ext, per_file_limit.min(remaining));
        let (category, confidence, explanation) = classify(relative, &ext, sort);
        let target = planned_target(&category, relative);
        let id = Uuid::new_v4().to_string();
        let mut context = AiFileContext {
            id: id.clone(),
            path: relative.to_string_lossy().into_owned(),
            extension: ext.clone(),
            size_bytes: metadata.len(),
            created_at: format_file_time(metadata.created().ok()),
            modified_at: format_file_time(metadata.modified().ok()),
            suggested_category: category.clone(),
            content_extract,
            content_status,
        };
        fit_context_into_model_budget(&mut context);
        total_chars = total_chars.saturating_add(
            context
                .content_extract
                .as_ref()
                .map_or(0, |text| text.chars().count()),
        );
        contexts.push(context);
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
    if inaccessible_files > 0 {
        warnings.push(format!(
            "Нет доступа к {inaccessible_files} объектам; их имена и пути скрыты."
        ));
    }
    Ok(PreparedAnalysis {
        items,
        contexts,
        total_chars,
        warnings,
        inaccessible_files,
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

fn emit_analysis_log(app: &tauri::AppHandle, event: AnalysisLogEvent) {
    let _ = app.emit("analysis-log", event);
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

fn extension_summary(contexts: &[AiFileContext]) -> Vec<ExtensionCount> {
    const MAX_EXTENSIONS: usize = 12;
    let mut counts = BTreeMap::<String, usize>::new();
    for context in contexts {
        let extension = if context.extension.trim().is_empty() {
            "без расширения".to_string()
        } else {
            format!(".{}", context.extension.trim().to_lowercase())
        };
        *counts.entry(extension).or_default() += 1;
    }
    let mut summary: Vec<ExtensionCount> = counts
        .into_iter()
        .map(|(extension, count)| ExtensionCount { extension, count })
        .collect();
    summary.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.extension.cmp(&right.extension))
    });
    if summary.len() > MAX_EXTENSIONS {
        let other_count = summary[MAX_EXTENSIONS..]
            .iter()
            .map(|entry| entry.count)
            .sum();
        summary.truncate(MAX_EXTENSIONS);
        summary.push(ExtensionCount {
            extension: "другие".into(),
            count: other_count,
        });
    }
    summary
}

fn model_context_bytes(context: &AiFileContext) -> usize {
    serde_json::to_vec(context)
        .map(|serialized| serialized.len())
        .unwrap_or(usize::MAX)
}

fn model_batch_context_bytes(batch: &[AiFileContext]) -> usize {
    batch.iter().fold(0usize, |total, context| {
        total.saturating_add(model_context_bytes(context))
    })
}

fn shorten_text(value: &str, maximum_chars: usize) -> String {
    value.chars().take(maximum_chars).collect()
}

fn fit_context_into_model_budget(context: &mut AiFileContext) {
    let mut was_shortened = false;
    while model_context_bytes(context) > MAX_BATCH_CONTEXT_BYTES {
        let Some(text) = context.content_extract.as_ref() else {
            break;
        };
        let length = text.chars().count();
        if length == 0 {
            break;
        }
        let reduced_length = length.saturating_mul(3).saturating_div(4).max(1);
        context.content_extract = Some(shorten_text(text, reduced_length));
        was_shortened = true;
    }
    if was_shortened {
        context.content_status =
            "Текст сокращён до безопасного размера ИИ-запроса; использованы также метаданные файла."
                .into();
    }
}

fn model_batch_ranges(contexts: &[AiFileContext]) -> Vec<Range<usize>> {
    if contexts.is_empty() {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut start = 0usize;
    let mut current_bytes = 0usize;
    for (index, context) in contexts.iter().enumerate() {
        let context_bytes = model_context_bytes(context);
        let current_count = index.saturating_sub(start);
        if current_count > 0
            && (current_count >= AI_BATCH_SIZE
                || current_bytes.saturating_add(context_bytes) > MAX_BATCH_CONTEXT_BYTES)
        {
            ranges.push(start..index);
            start = index;
            current_bytes = 0;
        }
        current_bytes = current_bytes.saturating_add(context_bytes);
    }
    ranges.push(start..contexts.len());
    ranges
}

fn anonymized_error(error: &str) -> (String, String) {
    let (kind, fallback) = if error.contains("Тайм-аут") || error.contains("timed out") {
        ("timeout", "Модель не ответила в установленное время")
    } else if error.contains("невалидный JSON") || error.contains("не является JSON")
    {
        ("invalid_json", "Ответ модели не является валидным JSON")
    } else if error.contains("HTTP") {
        ("http", "API модели вернул ошибку HTTP")
    } else if error.contains("choices[0].message.content") {
        (
            "response_shape",
            "В ответе модели отсутствует ожидаемое поле",
        )
    } else if error.contains("запрос") || error.contains("request") {
        ("network", "Сетевая ошибка запроса к модели")
    } else {
        ("unknown", "Неизвестная техническая ошибка модели")
    };
    let mut detail = error.trim().to_string();
    for marker in [" for url", "http://", "https://", "file://"] {
        if let Some(index) = detail.find(marker) {
            detail.truncate(index);
        }
    }
    if detail.contains("/Users/")
        || detail.contains("/Volumes/")
        || detail.contains("\\Users\\")
        || detail.contains("\\\\")
        || detail.as_bytes().windows(3).any(|part| {
            part[0].is_ascii_alphabetic() && part[1] == b':' && matches!(part[2], b'\\' | b'/')
        })
        || detail.is_empty()
    {
        detail = fallback.into();
    }
    detail = detail.trim_end_matches([' ', ':', '(', '-']).to_string();
    if detail.chars().count() > 240 {
        detail = detail.chars().take(237).collect::<String>() + "…";
    }
    (kind.into(), detail)
}

#[tauri::command]
fn apply_sort(folder: String, items: Vec<PlanItem>) -> Result<usize, String> {
    let root = canonical_root(&folder)?;
    ensure_root_writable(&root)?;
    let mut planned = Vec::new();
    let mut reserved_destinations = HashSet::new();
    let mut seen_sources = HashSet::new();
    for item in items {
        if !item.included {
            continue;
        }
        let source = canonical_inside(&root, Path::new(&item.source))?;
        let destination = safe_destination(&root, &item.target)?;
        if source == destination {
            continue;
        }
        if !seen_sources.insert(normalized_path_key(&source)) {
            return Err("Один исходный файл добавлен в план несколько раз".into());
        }
        let destination = conflict_free_reserved(&destination, &mut reserved_destinations);
        planned.push(PlannedMove {
            from: source,
            to: destination,
        });
    }
    if planned.is_empty() {
        return Ok(0);
    }

    let records: Vec<MoveRecord> = planned
        .iter()
        .map(|operation| MoveRecord {
            from: operation.from.to_string_lossy().into_owned(),
            to: operation.to.to_string_lossy().into_owned(),
        })
        .collect();
    let history_data = serde_json::to_vec_pretty(&records).map_err(|e| e.to_string())?;
    let completed = execute_moves(&planned)?;
    if let Err(error) = write_history(&root, &history_data) {
        return Err(operation_error_with_rollback(
            "Не удалось сохранить журнал отмены",
            &error,
            &completed,
        ));
    }
    Ok(completed.len())
}

#[tauri::command]
fn undo_last_sort(folder: String) -> Result<usize, String> {
    let root = canonical_root(&folder)?;
    let history = root.join(HISTORY_FILE);
    let raw = fs::read(&history).map_err(|_| "Нет операции для отмены".to_string())?;
    let records: Vec<MoveRecord> =
        serde_json::from_slice(&raw).map_err(|_| "Журнал операции повреждён".to_string())?;
    ensure_root_writable(&root)?;
    let mut planned = Vec::new();
    let mut reserved_destinations = HashSet::new();
    let mut seen_sources = HashSet::new();
    for record in records.into_iter().rev() {
        let current = canonical_inside(&root, Path::new(&record.to))?;
        let original = safe_recorded_destination(&root, Path::new(&record.from))?;
        if !seen_sources.insert(normalized_path_key(&current)) {
            return Err("Журнал отмены содержит повторяющийся файл".into());
        }
        let original = conflict_free_reserved(&original, &mut reserved_destinations);
        planned.push(PlannedMove {
            from: current,
            to: original,
        });
    }
    let completed = execute_moves(&planned)?;
    if let Err(error) = fs::remove_file(history) {
        return Err(operation_error_with_rollback(
            "Не удалось удалить использованный журнал отмены",
            &error.to_string(),
            &completed,
        ));
    }
    Ok(completed.len())
}

#[tauri::command]
async fn test_connection(ai: AiSettings) -> Result<String, String> {
    if ai.base_url.trim().is_empty() {
        return Err("Укажите базовый URL".into());
    }
    let url = test_connection_url(&ai);
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
        .map_err(|error| anonymized_error(&format!("Сервис не ответил: {error}")).1)?;
    response
        .error_for_status()
        .map_err(|error| anonymized_error(&format!("Сервис вернул ошибку HTTP: {error}")).1)?;
    Ok("Подключение успешно".into())
}

fn test_connection_url(ai: &AiSettings) -> String {
    let base_url = ai.base_url.trim_end_matches('/');
    if ai.provider == "ollama" {
        format!("{}/api/tags", base_url.trim_end_matches("/v1"))
    } else {
        format!("{base_url}/models")
    }
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
        .map_err(|error| anonymized_error(&format!("Не удалось получить модели: {error}")).1)?
        .error_for_status()
        .map_err(|error| anonymized_error(&format!("Сервис вернул ошибку HTTP: {error}")).1)?;
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
    let normalized = name.to_ascii_lowercase();
    !(entry.file_type().is_dir()
        && (name.eq_ignore_ascii_case(SORTED_DIR)
            || normalized.ends_with(".app")
            || normalized == "$recycle.bin"
            || normalized == "system volume information"))
}
fn skip_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    if name.starts_with(".ai-file-sorter-") {
        return true;
    }
    matches!(
        name.to_ascii_lowercase().as_str(),
        ".ds_store" | ".localized" | "thumbs.db" | "desktop.ini"
    )
}

fn is_loopback_endpoint(value: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(value.trim()) else {
        return false;
    };
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim_end_matches('.');
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
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
                Component::ParentDir
                    | Component::RootDir
                    | Component::Prefix(_)
                    | Component::CurDir
            )
        })
    {
        return Err("Недопустимый целевой путь".into());
    }
    for component in p.components() {
        let name = component.as_os_str().to_string_lossy();
        if !windows_compatible_component(&name) {
            return Err("Имя папки или файла несовместимо с Windows".into());
        }
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
        .any(|component| matches!(component, Component::ParentDir))
        || !candidate.starts_with(root)
    {
        return Err("Путь выходит за пределы выбранной папки".into());
    }
    Ok(candidate.to_path_buf())
}

fn normalized_path_key(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/").to_lowercase()
}

fn conflict_free_reserved(path: &Path, reserved: &mut HashSet<String>) -> PathBuf {
    let stem = path.file_stem().and_then(|x| x.to_str()).unwrap_or("file");
    let ext = path
        .extension()
        .and_then(|x| x.to_str())
        .map(|x| format!(".{x}"))
        .unwrap_or_default();
    for index in 1.. {
        let candidate = if index == 1 {
            path.to_path_buf()
        } else {
            path.with_file_name(format!("{stem} ({index}){ext}"))
        };
        let key = normalized_path_key(&candidate);
        if !candidate.exists() && reserved.insert(key) {
            return candidate;
        }
    }
    unreachable!()
}

fn ensure_root_writable(root: &Path) -> Result<(), String> {
    let probe = root.join(format!(".ai-file-sorter-write-test-{}", Uuid::new_v4()));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
        .map_err(|error| {
            format!(
                "Выбранная папка или диск недоступны для записи: {error}. Файлы не перемещались"
            )
        })?;
    drop(file);
    if let Err(error) = fs::remove_file(&probe) {
        return Err(format!(
            "Проверочный файл создан, но не удалён: {error}. Сортировка не запускалась"
        ));
    }
    Ok(())
}

fn execute_moves(planned: &[PlannedMove]) -> Result<Vec<PlannedMove>, String> {
    let mut completed = Vec::new();
    for operation in planned {
        if let Some(parent) = operation.to.parent() {
            if let Err(error) = fs::create_dir_all(parent) {
                return Err(operation_error_with_rollback(
                    "Не удалось создать целевую папку",
                    &error.to_string(),
                    &completed,
                ));
            }
        }
        if operation.to.exists() {
            return Err(operation_error_with_rollback(
                "Целевой файл появился после проверки конфликтов",
                "операция остановлена, чтобы не перезаписать существующий файл",
                &completed,
            ));
        }
        if let Err(error) = fs::rename(&operation.from, &operation.to) {
            return Err(operation_error_with_rollback(
                "Не удалось переместить файл",
                &error.to_string(),
                &completed,
            ));
        }
        completed.push(operation.clone());
    }
    Ok(completed)
}

fn rollback_moves(completed: &[PlannedMove]) -> Result<(), String> {
    let mut failures = 0usize;
    for operation in completed.iter().rev() {
        if operation.from.exists() || !operation.to.exists() {
            failures += 1;
            continue;
        }
        if let Some(parent) = operation.from.parent() {
            if fs::create_dir_all(parent).is_err() {
                failures += 1;
                continue;
            }
        }
        if fs::rename(&operation.to, &operation.from).is_err() {
            failures += 1;
        }
    }
    if failures == 0 {
        Ok(())
    } else {
        Err(format!("не удалось вернуть файлов: {failures}"))
    }
}

fn operation_error_with_rollback(context: &str, error: &str, completed: &[PlannedMove]) -> String {
    if completed.is_empty() {
        return format!("{context}: {error}. Файлы не перемещались");
    }
    match rollback_moves(completed) {
        Ok(()) => format!("{context}: {error}. Уже перемещённые файлы возвращены обратно"),
        Err(rollback_error) => format!(
            "{context}: {error}. ВНИМАНИЕ: автоматический откат выполнен не полностью ({rollback_error})"
        ),
    }
}

fn write_history(root: &Path, data: &[u8]) -> Result<(), String> {
    let history = root.join(HISTORY_FILE);
    let temporary = root.join(format!("{HISTORY_FILE}.{}.tmp", Uuid::new_v4()));
    let backup = root.join(format!("{HISTORY_FILE}.{}.bak", Uuid::new_v4()));
    fs::write(&temporary, data).map_err(io_error)?;

    let had_history = history.exists();
    if had_history {
        if let Err(error) = fs::rename(&history, &backup) {
            let _ = fs::remove_file(&temporary);
            return Err(error.to_string());
        }
    }
    if let Err(error) = fs::rename(&temporary, &history) {
        if had_history {
            let _ = fs::rename(&backup, &history);
        }
        let _ = fs::remove_file(&temporary);
        return Err(error.to_string());
    }
    if had_history {
        let _ = fs::remove_file(backup);
    }
    Ok(())
}

fn io_error(e: std::io::Error) -> String {
    e.to_string()
}

fn is_windows_reserved_name(value: &str) -> bool {
    let stem = value
        .split('.')
        .next()
        .unwrap_or(value)
        .to_ascii_uppercase();
    matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || stem
            .strip_prefix("COM")
            .or_else(|| stem.strip_prefix("LPT"))
            .is_some_and(|number| {
                matches!(number, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
            })
}

fn windows_compatible_component(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && !value.ends_with([' ', '.'])
        && !is_windows_reserved_name(value)
        && !value.chars().any(|ch| {
            ch.is_control() || matches!(ch, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|')
        })
}

fn safe_name(value: &str) -> String {
    let clean: String = value
        .chars()
        .map(|c| {
            if c.is_control() || matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') {
                '_'
            } else {
                c
            }
        })
        .take(80)
        .collect();
    let clean = clean.trim().trim_end_matches('.').trim_end().to_string();
    if clean.is_empty() || clean == "." || clean == ".." {
        "Прочее".into()
    } else if is_windows_reserved_name(&clean) {
        format!("_{clean}")
    } else {
        clean
    }
}
fn planned_target(category: &str, relative: &Path) -> PathBuf {
    let filename = relative
        .file_name()
        .and_then(|name| name.to_str())
        .map(safe_name)
        .unwrap_or_else(|| "файл".into());
    Path::new(SORTED_DIR)
        .join(safe_name(category))
        .join(filename)
}

fn restricted_category(value: &str, fallback: &str) -> String {
    let normalized = value.trim().to_lowercase();
    let category = if normalized.contains("работ")
        || normalized.contains("work")
        || normalized.contains("business")
    {
        Some("Работа")
    } else if normalized.contains("личн")
        || normalized.contains("personal")
        || normalized.contains("private")
    {
        Some("Личное")
    } else if normalized.contains("финанс")
        || normalized.contains("finance")
        || normalized.contains("bank")
    {
        Some("Финансы")
    } else if normalized.contains("учеб")
        || normalized.contains("study")
        || normalized.contains("education")
    {
        Some("Учёба")
    } else if normalized.contains("медиа")
        || normalized.contains("media")
        || normalized.contains("photo")
        || normalized.contains("video")
    {
        Some("Медиа")
    } else if normalized.contains("архив") || normalized.contains("archive") {
        Some("Архив")
    } else if normalized.contains("загруз")
        || normalized.contains("download")
        || normalized.contains("install")
    {
        Some("Загрузчики")
    } else if normalized.contains("проч")
        || normalized.contains("other")
        || normalized.contains("misc")
    {
        Some("Прочее")
    } else {
        None
    };
    let safe_fallback = STANDARD_CATEGORIES
        .iter()
        .copied()
        .find(|category| *category == fallback)
        .unwrap_or("Прочее");
    category.unwrap_or(safe_fallback).to_string()
}

fn bounded_custom_prompt(value: &str) -> String {
    value.chars().take(MAX_CUSTOM_PROMPT_CHARS).collect()
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
    let max_bytes = if limit == usize::MAX {
        u64::MAX
    } else {
        u64::try_from(limit.saturating_mul(4).saturating_add(4)).unwrap_or(u64::MAX)
    };
    let mut bytes = Vec::new();
    let read_result = fs::File::open(path).and_then(|file| {
        let mut reader = file.take(max_bytes);
        reader.read_to_end(&mut bytes)
    });
    match read_result {
        Ok(_) => {
            let text = String::from_utf8_lossy(&bytes);
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
fn classify(relative: &Path, ext: &str, _sort: &SortSettings) -> (String, f32, String) {
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
    let category = if installer {
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

async fn refine_with_model<P, L>(
    client: &reqwest::Client,
    ai: &AiSettings,
    sort: &SortSettings,
    items: &mut [PlanItem],
    contexts: &[AiFileContext],
    cancelled: Arc<AtomicBool>,
    observers: (P, L),
) -> Result<AiRefinement, String>
where
    P: FnMut(AnalysisProgress),
    L: FnMut(AnalysisLogEvent),
{
    let (progress, log_event) = observers;
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
        log_event,
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
    let context_bytes = model_batch_context_bytes(&batch);
    if context_bytes > MAX_BATCH_CONTEXT_BYTES {
        return Err(BatchError::Failure(format!(
            "Входной контекст пакета ({context_bytes} байт) превышает безопасный лимит {MAX_BATCH_CONTEXT_BYTES} байт."
        )));
    }
    let category_rule = "Категория должна быть ровно одной из: Работа, Личное, Финансы, Учёба, Медиа, Архив, Загрузчики, Прочее.";
    let instruction = if sort.mode == "custom" && !sort.custom_prompt.trim().is_empty() {
        format!(
            "{category_rule} Дополнительная инструкция пользователя: {}",
            bounded_custom_prompt(&sort.custom_prompt)
        )
    } else {
        format!("{category_rule} Установочные файлы с расширениями DMG, EXE, PKG, MSI и похожими всегда относятся к Загрузчикам.")
    };
    let url = chat_completions_url(ai);
    let prompt = format!("Классифицируй только этот небольшой пакет файлов. Инструкция: {instruction}\nДля каждого файла сначала используй contentExtract, если он есть. Если его нет, анализируй только метаданные: path, extension, sizeBytes, даты и suggestedCategory. Не выдумывай содержимое. Верни ТОЛЬКО компактный JSON-массив объектов {{id, category, confidence}}. Без explanation, Markdown и любого текста вне JSON. Верни ровно одно решение для каждого переданного id.\nФайлы: {}", serde_json::to_string(&batch).map_err(|error| BatchError::Failure(error.to_string()))?);
    let body = serde_json::json!({"model":ai.model,"temperature":0,"max_tokens":MAX_MODEL_RESPONSE_TOKENS,"messages":[{"role":"system","content":"Отвечай только компактным валидным JSON-массивом. Каждый объект: id, category, confidence. Никакого Markdown, explanation или текста вне JSON."},{"role":"user","content":prompt}]});
    let mut request = client.post(&url).json(&body);
    if !ai.api_key.trim().is_empty() {
        request = request.bearer_auth(&ai.api_key);
    }
    let operation =
        async {
            let response = request.send().await.map_err(|error| {
                BatchError::Failure(format!("Ошибка запроса к модели: {error}"))
            })?;
            let status = response.status();
            if !status.is_success() {
                let response_body = response.text().await.unwrap_or_default();
                return Err(BatchError::Failure(model_http_error(
                    status,
                    &response_body,
                )));
            }
            let value: serde_json::Value = response.json().await.map_err(|error| {
                BatchError::Failure(format!("Ответ API не является JSON: {error}"))
            })?;
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

fn chat_completions_url(ai: &AiSettings) -> String {
    let base_url = ai.base_url.trim_end_matches('/');
    if ai.provider == "ollama" && !base_url.ends_with("/v1") {
        format!("{base_url}/v1/chat/completions")
    } else {
        format!("{base_url}/chat/completions")
    }
}

fn model_http_error(status: reqwest::StatusCode, response_body: &str) -> String {
    let body = response_body.to_lowercase();
    let reason = if ["context", "token", "max_tokens", "context_length"]
        .iter()
        .any(|marker| body.contains(marker))
    {
        "запрос превышает доступный контекст модели"
    } else if [
        "payload",
        "request too large",
        "body too large",
        "too large",
    ]
    .iter()
    .any(|marker| body.contains(marker))
    {
        "запрос слишком большой для API модели"
    } else if ["model not found", "unknown model", "model is not loaded"]
        .iter()
        .any(|marker| body.contains(marker))
    {
        "выбранная модель не найдена или не загружена"
    } else if ["api key", "unauthorized", "authentication", "forbidden"]
        .iter()
        .any(|marker| body.contains(marker))
    {
        "API отклонил авторизацию"
    } else {
        "API отклонил запрос; подробности ответа скрыты для конфиденциальности"
    };
    format!("Модель вернула HTTP {status}: {reason}.")
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

#[allow(clippy::too_many_arguments)]
fn batch_log_event(
    phase: &str,
    attempt: usize,
    batch_number: usize,
    total_batches: usize,
    batch: &[AiFileContext],
    started: Instant,
    outcome: &str,
    successful_files: usize,
    unresolved_files: usize,
    error_kind: Option<String>,
    error_detail: Option<String>,
) -> AnalysisLogEvent {
    AnalysisLogEvent {
        phase: phase.into(),
        attempt: Some(attempt),
        batch_number: Some(batch_number),
        total_batches: Some(total_batches),
        file_count: batch.len(),
        extensions: extension_summary(batch),
        duration_ms: elapsed_ms(started),
        outcome: outcome.into(),
        successful_files,
        unresolved_files,
        skipped_files: 0,
        input_bytes: Some(model_batch_context_bytes(batch)),
        error_kind,
        error_detail,
    }
}

async fn refine_in_batches<F, Fut, P, L>(
    sort: &SortSettings,
    items: &mut [PlanItem],
    contexts: &[AiFileContext],
    mut request_batch: F,
    mut progress: P,
    mut log_event: L,
) -> Result<AiRefinement, String>
where
    F: FnMut(Vec<AiFileContext>) -> Fut,
    Fut: Future<Output = Result<Vec<AiDecision>, BatchError>>,
    P: FnMut(AnalysisProgress),
    L: FnMut(AnalysisLogEvent),
{
    if items.is_empty() {
        progress(AnalysisProgress {
            phase: "complete".into(),
            completed_batches: 0,
            total_batches: 0,
            processed_files: 0,
            pending_files: 0,
            not_attempted_files: 0,
            retry_pending_files: 0,
            message: "В выбранной папке нет файлов для анализа.".into(),
        });
        return Ok(AiRefinement::default());
    }
    let mut result = AiRefinement::default();
    let mut first_failures = HashMap::<String, String>::new();
    let main_batch_ranges = model_batch_ranges(contexts);
    let main_batches = main_batch_ranges.len();
    let mut attempted_main_files = 0usize;

    for (index, range) in main_batch_ranges.iter().enumerate() {
        let batch = &contexts[range.clone()];
        let not_attempted_before = contexts.len().saturating_sub(attempted_main_files);
        progress(AnalysisProgress {
            phase: "main".into(),
            completed_batches: index,
            total_batches: main_batches,
            processed_files: processed_count(items),
            pending_files: not_attempted_before + first_failures.len(),
            not_attempted_files: not_attempted_before,
            retry_pending_files: first_failures.len(),
            message: format!("Основной проход: пакет {} из {main_batches}…", index + 1),
        });
        let batch_started = Instant::now();
        match request_batch(batch.to_vec()).await {
            Ok(decisions) => {
                let missing = apply_batch_decisions(sort, items, batch, decisions);
                let successful = batch.len() - missing.len();
                if !missing.is_empty() {
                    let reason = "Модель вернула частичный ответ: решение для файла отсутствует."
                        .to_string();
                    for id in &missing {
                        first_failures.insert(id.clone(), reason.clone());
                    }
                    result.warnings.push(format!("Основной проход, пакет {}: модель не вернула решения для {} файлов; они отправлены на повторную попытку.", index + 1, missing.len()));
                }
                log_event(batch_log_event(
                    "main",
                    1,
                    index + 1,
                    main_batches,
                    batch,
                    batch_started,
                    if missing.is_empty() {
                        "success"
                    } else {
                        "partial"
                    },
                    successful,
                    missing.len(),
                    (!missing.is_empty()).then(|| "partial_response".into()),
                    (!missing.is_empty())
                        .then(|| "Модель не вернула решения для части файлов пакета".into()),
                ));
            }
            Err(BatchError::Failure(error)) => {
                let (error_kind, error_detail) = anonymized_error(&error);
                for context in batch {
                    first_failures.insert(context.id.clone(), error_detail.clone());
                }
                result.warnings.push(format!("Основной проход, пакет {}: {error_detail}. На повторную попытку отправлено {} файлов.", index + 1, batch.len()));
                log_event(batch_log_event(
                    "main",
                    1,
                    index + 1,
                    main_batches,
                    batch,
                    batch_started,
                    "error",
                    0,
                    batch.len(),
                    Some(error_kind),
                    Some(error_detail),
                ));
            }
            Err(BatchError::Cancelled) => {
                log_event(batch_log_event(
                    "main",
                    1,
                    index + 1,
                    main_batches,
                    batch,
                    batch_started,
                    "cancelled",
                    0,
                    batch.len(),
                    Some("cancelled".into()),
                    Some("Анализ отменён пользователем".into()),
                ));
                return Err("Анализ отменён пользователем".into());
            }
        }
        attempted_main_files = attempted_main_files.saturating_add(batch.len());
        let not_attempted_after = contexts.len().saturating_sub(attempted_main_files);
        progress(AnalysisProgress {
            phase: "main".into(),
            completed_batches: index + 1,
            total_batches: main_batches,
            processed_files: processed_count(items),
            pending_files: not_attempted_after + first_failures.len(),
            not_attempted_files: not_attempted_after,
            retry_pending_files: first_failures.len(),
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
    let retry_batch_ranges = model_batch_ranges(&retry_contexts);
    let retry_batches = retry_batch_ranges.len();
    for (index, range) in retry_batch_ranges.iter().enumerate() {
        let batch = &retry_contexts[range.clone()];
        progress(AnalysisProgress {
            phase: "retry".into(),
            completed_batches: index,
            total_batches: retry_batches,
            processed_files: processed_count(items),
            pending_files: retry_pending_count(items),
            not_attempted_files: 0,
            retry_pending_files: retry_pending_count(items),
            message: format!("Повторный проход: пакет {} из {retry_batches}…", index + 1),
        });
        let batch_started = Instant::now();
        match request_batch(batch.to_vec()).await {
            Ok(decisions) => {
                let missing = apply_batch_decisions(sort, items, batch, decisions);
                let successful = batch.len() - missing.len();
                result.summary.retry_succeeded += successful;
                if !missing.is_empty() {
                    let reason =
                        "Модель снова вернула частичный ответ: решение для файла отсутствует.";
                    for id in &missing {
                        mark_unprocessed(items, id, first_failures.get(id), reason);
                    }
                    result.warnings.push(format!("Повторный проход, пакет {}: для {} файлов снова нет решения; они направлены в «{}».", index + 1, missing.len(), UNPROCESSED_CATEGORY));
                }
                log_event(batch_log_event(
                    "retry",
                    2,
                    index + 1,
                    retry_batches,
                    batch,
                    batch_started,
                    if missing.is_empty() {
                        "success"
                    } else {
                        "partial"
                    },
                    successful,
                    missing.len(),
                    (!missing.is_empty()).then(|| "partial_response".into()),
                    (!missing.is_empty()).then(|| {
                        "Модель повторно не вернула решения для части файлов пакета".into()
                    }),
                ));
            }
            Err(BatchError::Failure(error)) => {
                let (error_kind, error_detail) = anonymized_error(&error);
                for context in batch {
                    mark_unprocessed(
                        items,
                        &context.id,
                        first_failures.get(&context.id),
                        &error_detail,
                    );
                }
                result.warnings.push(format!(
                    "Повторный проход, пакет {}: {error_detail}. В «{}» направлено {} файлов.",
                    index + 1,
                    UNPROCESSED_CATEGORY,
                    batch.len()
                ));
                log_event(batch_log_event(
                    "retry",
                    2,
                    index + 1,
                    retry_batches,
                    batch,
                    batch_started,
                    "error",
                    0,
                    batch.len(),
                    Some(error_kind),
                    Some(error_detail),
                ));
            }
            Err(BatchError::Cancelled) => {
                log_event(batch_log_event(
                    "retry",
                    2,
                    index + 1,
                    retry_batches,
                    batch,
                    batch_started,
                    "cancelled",
                    0,
                    batch.len(),
                    Some("cancelled".into()),
                    Some("Анализ отменён пользователем".into()),
                ));
                return Err("Анализ отменён пользователем".into());
            }
        }
        progress(AnalysisProgress {
            phase: "retry".into(),
            completed_batches: index + 1,
            total_batches: retry_batches,
            processed_files: processed_count(items),
            pending_files: retry_pending_count(items),
            not_attempted_files: 0,
            retry_pending_files: retry_pending_count(items),
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
        not_attempted_files: 0,
        retry_pending_files: 0,
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
        let category = restricted_category(&decision.category, &item.category);
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
    item.target = planned_target(category, Path::new(&item.relative_path))
        .to_string_lossy()
        .into_owned();
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
                target: format!("{SORTED_DIR}/Личное/file-{index}.txt"),
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
        assert_eq!(items[5].ai_status, AiStatus::Processed);
    }

    #[tokio::test]
    async fn second_failure_moves_only_failed_files_to_unprocessed() {
        let (mut items, contexts) = plan(20);
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
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(
            result.summary,
            AiSummary {
                ai_processed: 10,
                retry_succeeded: 0,
                ai_unprocessed: 10
            }
        );
        assert!(items[..10]
            .iter()
            .all(|item| item.ai_status == AiStatus::Processed));
        for item in &items[10..] {
            assert_eq!(item.ai_status, AiStatus::Unprocessed);
            assert_eq!(item.category, UNPROCESSED_CATEGORY);
            let expected_parent = PathBuf::from(SORTED_DIR).join(UNPROCESSED_CATEGORY);
            assert!(Path::new(&item.target).starts_with(expected_parent));
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
                    Ok(decisions(&batch[..3]))
                } else {
                    Ok(decisions(&batch))
                };
                std::future::ready(response)
            },
            |_| {},
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(calls, 3);
        assert_eq!(
            result.summary,
            AiSummary {
                ai_processed: 12,
                retry_succeeded: 7,
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

    #[test]
    fn string_confidence_does_not_discard_a_valid_model_decision() {
        let value = serde_json::json!({
            "choices": [{"message": {"content": r#"[{"id":"file-1","category":"Работа","confidence":"high"}]"#}}]
        });
        let decisions = parse_model_response(&value).unwrap();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].confidence, Some(0.75));
    }

    #[test]
    fn diagnostic_errors_hide_urls_and_local_paths() {
        let (_, url_detail) = anonymized_error(
            "Модель вернула ошибку HTTP: 400 Bad Request for url (http://127.0.0.1:1234/v1/chat/completions)",
        );
        assert!(url_detail.contains("400 Bad Request"));
        assert!(!url_detail.contains("http://"));

        let (_, path_detail) =
            anonymized_error("Ошибка запроса к модели: /Users/private/Documents/secret-file.txt");
        assert_eq!(path_detail, "Сетевая ошибка запроса к модели");
        assert!(!path_detail.contains("secret-file"));
    }

    #[test]
    fn context_error_is_useful_without_exposing_server_response() {
        let error = model_http_error(
            reqwest::StatusCode::BAD_REQUEST,
            r#"{"error":{"message":"context length 12000 exceeds 4096 for /Users/private/secret.txt"}}"#,
        );
        assert!(error.contains("превышает доступный контекст"));
        assert!(!error.contains("secret.txt"));
        assert!(!error.contains("/Users/"));
    }

    #[test]
    fn ollama_uses_its_openai_compatible_chat_endpoint() {
        let ai = AiSettings {
            provider: "ollama".into(),
            base_url: "http://127.0.0.1:11434".into(),
            model: "test".into(),
            api_key: String::new(),
            cloud_consent: false,
        };
        assert_eq!(
            chat_completions_url(&ai),
            "http://127.0.0.1:11434/v1/chat/completions"
        );
        assert_eq!(test_connection_url(&ai), "http://127.0.0.1:11434/api/tags");
    }

    #[test]
    fn contexts_are_split_by_payload_size_and_file_count() {
        let (_, mut contexts) = plan(5);
        for context in &mut contexts {
            context.content_extract = Some("a".repeat(3_800));
            fit_context_into_model_budget(context);
        }
        let ranges = model_batch_ranges(&contexts);
        assert!(ranges.len() > 1);
        for range in ranges {
            let batch = &contexts[range];
            assert!(batch.len() <= AI_BATCH_SIZE);
            assert!(model_batch_context_bytes(batch) <= MAX_BATCH_CONTEXT_BYTES);
        }
    }

    #[test]
    fn oversized_context_text_is_trimmed_before_request() {
        let (_, mut contexts) = plan(1);
        contexts[0].content_extract = Some("a".repeat(MAX_BATCH_CONTEXT_BYTES * 2));
        fit_context_into_model_budget(&mut contexts[0]);
        assert!(model_context_bytes(&contexts[0]) <= MAX_BATCH_CONTEXT_BYTES);
        assert!(contexts[0].content_status.contains("сокращён"));
    }

    #[test]
    fn extension_summary_contains_no_file_names_or_paths() {
        let (_, mut contexts) = plan(3);
        contexts[0].extension = "PDF".into();
        contexts[1].extension = "pdf".into();
        contexts[2].extension.clear();
        let summary = extension_summary(&contexts);
        assert_eq!(
            summary,
            vec![
                ExtensionCount {
                    extension: ".pdf".into(),
                    count: 2,
                },
                ExtensionCount {
                    extension: "без расширения".into(),
                    count: 1,
                },
            ]
        );
        let serialized = serde_json::to_string(&summary).unwrap();
        assert!(!serialized.contains("исходная"));
        assert!(!serialized.contains("file-"));
    }

    #[tokio::test]
    async fn batch_diagnostics_report_attempts_extensions_and_separate_counts() {
        let (mut items, mut contexts) = plan(12);
        contexts[0].extension = "pdf".into();
        contexts[1].extension = "pdf".into();
        let mut progress_events = Vec::new();
        let mut log_events = Vec::new();
        refine_in_batches(
            &test_sort(),
            &mut items,
            &contexts,
            |batch| std::future::ready(Ok(decisions(&batch))),
            |progress| progress_events.push(progress),
            |event| log_events.push(event),
        )
        .await
        .unwrap();

        assert_eq!(log_events.len(), 2);
        assert_eq!(log_events[0].attempt, Some(1));
        assert_eq!(log_events[0].batch_number, Some(1));
        assert_eq!(log_events[0].file_count, 10);
        assert_eq!(log_events[0].outcome, "success");
        assert_eq!(log_events[0].successful_files, 10);
        assert!(log_events[0].extensions.contains(&ExtensionCount {
            extension: ".pdf".into(),
            count: 2,
        }));

        let after_first_batch = progress_events
            .iter()
            .find(|progress| progress.phase == "main" && progress.completed_batches == 1)
            .unwrap();
        assert_eq!(after_first_batch.processed_files, 10);
        assert_eq!(after_first_batch.not_attempted_files, 2);
        assert_eq!(after_first_batch.retry_pending_files, 0);
        assert_eq!(after_first_batch.pending_files, 2);
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
        fs::create_dir_all(root.join("$RECYCLE.BIN/trash")).unwrap();
        fs::create_dir_all(root.join("System Volume Information/index")).unwrap();
        fs::write(root.join("keep.txt"), "keep").unwrap();
        fs::write(root.join("Thumbs.db"), "skip").unwrap();
        fs::write(root.join("desktop.ini"), "skip").unwrap();
        fs::write(root.join(HISTORY_FILE), "skip").unwrap();
        fs::write(root.join(".ai-file-sorter-last-operation.test.bak"), "skip").unwrap();
        fs::write(root.join("AI Sorted/old/skip.txt"), "skip").unwrap();
        fs::write(root.join("Example.app/Contents/skip.txt"), "skip").unwrap();
        fs::write(root.join("$RECYCLE.BIN/trash/skip.txt"), "skip").unwrap();
        fs::write(
            root.join("System Volume Information/index/skip.txt"),
            "skip",
        )
        .unwrap();
        let files: Vec<PathBuf> = WalkDir::new(&root)
            .into_iter()
            .filter_entry(scan_entry)
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .filter(|entry| !skip_file(entry.path()))
            .map(|entry| entry.into_path())
            .collect();
        assert_eq!(files, vec![root.join("keep.txt")]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn names_are_sanitized() {
        assert_eq!(safe_name("A/B: C"), "A_B_ C");
        assert_eq!(safe_name("CON"), "_CON");
        assert_eq!(safe_name("nul.txt"), "_nul.txt");
        assert_eq!(safe_name("Категория.   "), "Категория");
        assert_eq!(safe_name(".."), "Прочее");
        assert!(safe_name(&"a".repeat(200)).chars().count() <= 80);
    }

    #[test]
    fn windows_incompatible_target_components_are_rejected() {
        let root = Path::new("/tmp/root");
        assert!(safe_destination(root, "AI Sorted/CON/file.txt").is_err());
        assert!(safe_destination(root, "AI Sorted/Работа./file.txt").is_err());
        assert!(safe_destination(root, "AI Sorted/Работа/file?.txt").is_err());
    }

    #[test]
    fn only_real_loopback_hosts_bypass_cloud_consent() {
        assert!(is_loopback_endpoint("http://localhost:1234/v1"));
        assert!(is_loopback_endpoint("http://127.0.0.1:11434"));
        assert!(is_loopback_endpoint("http://127.12.34.56:11434"));
        assert!(is_loopback_endpoint("http://[::1]:1234/v1"));
        assert!(!is_loopback_endpoint("https://example.com/?next=localhost"));
        assert!(!is_loopback_endpoint("https://localhost.example.com/v1"));
        assert!(!is_loopback_endpoint("not a URL containing localhost"));
    }

    #[test]
    fn planned_destinations_are_reserved_case_insensitively() {
        let mut reserved = HashSet::new();
        let root = std::env::temp_dir().join(format!("ai-file-sorter-case-{}", Uuid::new_v4()));
        let first = conflict_free_reserved(&root.join("Report.txt"), &mut reserved);
        let second = conflict_free_reserved(&root.join("report.txt"), &mut reserved);
        assert_eq!(first.file_name().unwrap(), "Report.txt");
        assert_eq!(second.file_name().unwrap(), "report (2).txt");
    }

    #[cfg(windows)]
    #[test]
    fn windows_absolute_history_paths_stay_inside_root() {
        let root = Path::new(r"C:\Selected");
        assert_eq!(
            safe_recorded_destination(root, Path::new(r"C:\Selected\incoming\file.txt")).unwrap(),
            PathBuf::from(r"C:\Selected\incoming\file.txt")
        );
        assert!(safe_recorded_destination(root, Path::new(r"D:\outside\file.txt")).is_err());
    }

    #[test]
    fn apply_rolls_back_files_when_a_later_move_fails() {
        let root = std::env::temp_dir().join(format!("ai-file-sorter-rollback-{}", Uuid::new_v4()));
        let first = root.join("incoming/first.txt");
        let second = root.join("incoming/second.txt");
        fs::create_dir_all(first.parent().unwrap()).unwrap();
        fs::write(&first, "first").unwrap();
        fs::write(&second, "second").unwrap();
        fs::write(root.join("blocked"), "not a directory").unwrap();
        let items = vec![
            PlanItem {
                id: "first".into(),
                source: first.to_string_lossy().into_owned(),
                relative_path: "incoming/first.txt".into(),
                target: "AI Sorted/Работа/first.txt".into(),
                category: "Работа".into(),
                explanation: "test".into(),
                confidence: 1.0,
                included: true,
                warning: None,
                ai_status: AiStatus::Processed,
                ai_error: None,
            },
            PlanItem {
                id: "second".into(),
                source: second.to_string_lossy().into_owned(),
                relative_path: "incoming/second.txt".into(),
                target: "blocked/sub/second.txt".into(),
                category: "Работа".into(),
                explanation: "test".into(),
                confidence: 1.0,
                included: true,
                warning: None,
                ai_status: AiStatus::Processed,
                ai_error: None,
            },
        ];
        let error = apply_sort(root.to_string_lossy().into_owned(), items).unwrap_err();
        assert!(error.contains("возвращены обратно"), "{error}");
        assert_eq!(fs::read_to_string(&first).unwrap(), "first");
        assert_eq!(fs::read_to_string(&second).unwrap(), "second");
        assert!(!root.join(HISTORY_FILE).exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn undo_rolls_back_restores_when_a_later_restore_fails() {
        let root =
            std::env::temp_dir().join(format!("ai-file-sorter-undo-rollback-{}", Uuid::new_v4()));
        let current_first = root.join("AI Sorted/first.txt");
        let current_second = root.join("AI Sorted/second.txt");
        fs::create_dir_all(current_first.parent().unwrap()).unwrap();
        fs::write(&current_first, "first").unwrap();
        fs::write(&current_second, "second").unwrap();
        fs::write(root.join("blocked"), "not a directory").unwrap();
        let canonical = fs::canonicalize(&root).unwrap();
        let original_first = canonical.join("incoming/first.txt");
        let original_second = canonical.join("blocked/sub/second.txt");
        let records = vec![
            MoveRecord {
                from: original_second.to_string_lossy().into_owned(),
                to: current_second.to_string_lossy().into_owned(),
            },
            MoveRecord {
                from: original_first.to_string_lossy().into_owned(),
                to: current_first.to_string_lossy().into_owned(),
            },
        ];
        fs::write(
            root.join(HISTORY_FILE),
            serde_json::to_vec(&records).unwrap(),
        )
        .unwrap();

        let error = undo_last_sort(root.to_string_lossy().into_owned()).unwrap_err();
        assert!(error.contains("возвращены обратно"), "{error}");
        assert_eq!(fs::read_to_string(&current_first).unwrap(), "first");
        assert_eq!(fs::read_to_string(&current_second).unwrap(), "second");
        assert!(!original_first.exists());
        assert!(root.join(HISTORY_FILE).exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn standard_categories_work() {
        assert_eq!(
            classify(Path::new("tax_invoice.pdf"), "pdf", &test_sort()).0,
            "Финансы"
        );
    }

    #[test]
    fn model_categories_cannot_create_new_top_level_folders() {
        assert_eq!(restricted_category("Business", "Личное"), "Работа");
        assert_eq!(restricted_category("Adello Visuals", "Личное"), "Личное");
        assert_eq!(
            restricted_category("unknown", "Кастомная категория"),
            "Прочее"
        );
    }

    #[test]
    fn custom_instruction_has_a_hard_character_limit() {
        let prompt = "я".repeat(MAX_CUSTOM_PROMPT_CHARS + 1);
        let bounded = bounded_custom_prompt(&prompt);
        assert_eq!(bounded.chars().count(), MAX_CUSTOM_PROMPT_CHARS);
    }

    #[test]
    fn planned_target_uses_only_category_and_filename() {
        assert_eq!(
            planned_target("Работа", Path::new("old/nested/report:final.pdf")),
            PathBuf::from("AI Sorted/Работа/report_final.pdf")
        );
    }

    #[test]
    fn text_preview_respects_character_limit_without_reading_the_whole_file() {
        let path = std::env::temp_dir().join(format!("ai-file-sorter-text-{}.txt", Uuid::new_v4()));
        fs::write(&path, "😀😀😀😀😀секретный хвост").unwrap();
        let (preview, status) = read_text_preview(&path, "txt", 5);
        assert_eq!(preview.as_deref(), Some("😀😀😀😀😀"));
        assert!(status.contains("фрагмент"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn unlimited_mode_uses_the_custom_per_file_text_limit() {
        let root =
            std::env::temp_dir().join(format!("ai-file-sorter-unlimited-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("large.txt"), "a".repeat(5_000)).unwrap();
        let sort = SortSettings {
            mode: "standard".into(),
            custom_prompt: String::new(),
            text_limit: 1_200,
            total_limit: usize::MAX,
            unlimited: true,
        };
        let prepared = prepare_analysis(
            root.to_string_lossy().into_owned(),
            &sort,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(prepared.contexts.len(), 1);
        assert!(prepared.contexts[0]
            .content_extract
            .as_ref()
            .is_some_and(|text| text.chars().count() == 1_200));
        assert!(prepared
            .warnings
            .iter()
            .any(|warning| warning.contains("Без общего лимита")));
        fs::remove_dir_all(root).unwrap();
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
