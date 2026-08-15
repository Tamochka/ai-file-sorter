import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { open } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { AiSettings, AnalysisLogEvent, AnalysisProgress, AnalysisResult, PlanItem, SortSettings, Status } from "./types";
import "./styles.css";

const initialAi: AiSettings = { provider: "lmstudio", baseUrl: "http://127.0.0.1:1234/v1", model: "", apiKey: "", cloudConsent: false };
const initialSort: SortSettings = { mode: "standard", customPrompt: "", textLimit: 12000, totalLimit: 500000, unlimited: false };
type ModelList = { models: string[]; activeModel?: string };
const aiStatusLabels = { processed: "Успешно обработан ИИ", retry_pending: "Ожидает повторной попытки", unprocessed: "Не обработан ИИ" } as const;
const timestamped = (message: string) => `[${new Date().toLocaleTimeString("ru-RU", { hour12: false })}] ${message}`;
const formatDuration = (milliseconds: number) => milliseconds < 1000 ? `${milliseconds} мс` : `${(milliseconds / 1000).toLocaleString("ru-RU", { maximumFractionDigits: 1 })} с`;
const formatExtensions = (event: AnalysisLogEvent) => event.extensions.length ? event.extensions.map(item => `${item.extension}×${item.count}`).join(", ") : "нет";
const formatAnalysisLog = (event: AnalysisLogEvent) => {
  if (event.phase === "scanning") return `Сканирование завершено · ${event.fileCount} файлов · расширения: ${formatExtensions(event)} · ${formatDuration(event.durationMs)}${event.skippedFiles ? ` · без доступа: ${event.skippedFiles}` : ""}`;
  const phase = event.phase === "main" ? "Основной проход" : "Повторный проход";
  const outcome = event.outcome === "success" ? "успешно" : event.outcome === "partial" ? "частичный ответ" : event.outcome === "cancelled" ? "отменено" : `ошибка${event.errorKind ? `: ${event.errorKind}` : ""}`;
  const unresolvedLabel = event.phase === "main" ? "на повтор" : "не обработано";
  return `${phase} ${event.batchNumber}/${event.totalBatches} · попытка ${event.attempt}/2 · ${event.fileCount} файлов · расширения: ${formatExtensions(event)} · ${formatDuration(event.durationMs)} · ${outcome} · успешно: ${event.successfulFiles} · ${unresolvedLabel}: ${event.unresolvedFiles}${event.errorDetail ? ` · ${event.errorDetail}` : ""}`;
};
const isBatchWarning = (warning: string) => warning.startsWith("Основной проход, пакет ") || warning.startsWith("Повторный проход, пакет ");
const isLoopbackEndpoint = (value: string) => {
  try {
    const host = new URL(value).hostname.replace(/^\[|\]$/g, "").replace(/\.$/, "").toLowerCase();
    return host === "localhost" || host === "::1" || /^127(?:\.\d{1,3}){3}$/.test(host);
  } catch {
    return false;
  }
};
const anonymizedUiError = (value: unknown) => {
  const text = String(value);
  if (/https?:\/\/|file:\/\/|\/Users\/|\/Volumes\/|\\Users\\|[A-Z]:[\\/]|\\\\/i.test(text)) return "Техническая ошибка; адрес или локальный путь скрыт.";
  return text.length > 240 ? `${text.slice(0, 237)}…` : text;
};

function App() {
  const [folder, setFolder] = useState(""); const [ai, setAi] = useState(initialAi); const [sort, setSort] = useState(initialSort);
  const [status, setStatus] = useState<Status>("Готово"); const [log, setLog] = useState<string[]>([timestamped("Выберите папку для начала работы.")]);
  const [result, setResult] = useState<AnalysisResult | null>(null); const [busy, setBusy] = useState(false); const [dark, setDark] = useState(() => localStorage.getItem("theme") === "dark");
  const [models, setModels] = useState<string[]>([]);
  const [analyzing, setAnalyzing] = useState(false); const [cancelRequested, setCancelRequested] = useState(false);
  const [progress, setProgress] = useState<AnalysisProgress | null>(null);
  const userSelectedModel = useRef(false);
  useEffect(() => { document.documentElement.dataset.theme = dark ? "dark" : "light"; }, [dark]);
  useEffect(() => { const subscription = listen<AnalysisProgress>("analysis-progress", event => setProgress(event.payload)); return () => { void subscription.then(unlisten => unlisten()); }; }, []);
  useEffect(() => { const subscription = listen<AnalysisLogEvent>("analysis-log", event => setLog(current => [...current, timestamped(formatAnalysisLog(event.payload))])); return () => { void subscription.then(unlisten => unlisten()); }; }, []);
  const included = useMemo(() => result?.items.filter((item) => item.included).length ?? 0, [result]);
  const locked = busy || analyzing;
  const local = isLoopbackEndpoint(ai.baseUrl);
  const writeLog = (message: string) => setLog((current) => [...current, timestamped(message)]);
  const loadModels = async (settings = ai) => { try { const found = await invoke<ModelList>("list_models", { ai: settings }); setModels(found.models); setAi(current => current.provider === settings.provider && current.baseUrl === settings.baseUrl && !userSelectedModel.current ? { ...current, model: found.activeModel || found.models[0] } : current); writeLog(found.activeModel ? "Активная модель LM Studio выбрана." : `Найдены модели: ${found.models.length}.`); } catch (error) { setModels([]); writeLog(`Не удалось получить список моделей: ${anonymizedUiError(error)}`); } };
  useEffect(() => { if (ai.provider === "lmstudio" || ai.provider === "ollama") void loadModels(); }, [ai.provider]);
  const changeProvider = (provider: AiSettings["provider"]) => { const baseUrl = provider === "ollama" ? "http://127.0.0.1:11434" : provider === "lmstudio" ? "http://127.0.0.1:1234/v1" : ai.baseUrl; userSelectedModel.current = false; setModels([]); setAi({ ...ai, provider, baseUrl, model: "" }); };
  const chooseFolder = async () => { try { const selected = await open({ directory: true, multiple: false, title: "Выберите папку" }); if (typeof selected === "string") { setFolder(selected); writeLog("Папка выбрана; путь скрыт в диагностическом логе."); } else { writeLog("Выбор папки отменён."); } } catch (error) { setStatus("Ошибка"); writeLog(`Не удалось открыть диалог выбора папки: ${anonymizedUiError(error)}`); } };
  const toggleTheme = () => { const next = !dark; setDark(next); localStorage.setItem("theme", next ? "dark" : "light"); };
  const analyze = async () => {
    if (!folder) return writeLog("Ошибка: сначала выберите папку.");
    if (!ai.model.trim()) return writeLog("Ошибка: укажите имя модели.");
    if (!local && !ai.cloudConsent) return writeLog("Нужно подтвердить передачу данных в облачный сервис.");
    setAnalyzing(true); setCancelRequested(false); setProgress(null); setStatus("Идёт анализ"); writeLog(`Запуск анализа · пакет: 10 файлов · максимум попыток: 2 · режим: ${sort.mode === "standard" ? "стандартный" : "кастомный"} · провайдер: ${ai.provider}`); writeLog("Сканирование файлов и подготовка безопасного плана…");
    try { const data = await invoke<AnalysisResult>("analyze_folder", { folder, ai, sort }); setResult(data); setStatus("Требуется подтверждение"); writeLog(`План готов: ${data.totalFiles} файлов, приблизительно ${data.estimatedChars.toLocaleString("ru-RU")} символов.`); writeLog(`Итог ИИ: обработано ${data.summary.aiProcessed}, успешно обработано повторно ${data.summary.retrySucceeded}, направлено в «Не обработано ИИ» ${data.summary.aiUnprocessed}.`); data.warnings.filter(warning => !isBatchWarning(warning)).forEach(writeLog); }
    catch (error) { const message = anonymizedUiError(error); if (message.includes("отменён пользователем")) { setStatus("Готово"); writeLog("Анализ отменён. Файлы не перемещались."); } else { setStatus("Ошибка"); writeLog(`Ошибка анализа: ${message}`); } } finally { setAnalyzing(false); setCancelRequested(false); }
  };
  const cancelAnalysis = async () => { setCancelRequested(true); writeLog("Запрошена отмена анализа…"); try { await invoke<boolean>("cancel_analysis"); } catch (error) { setCancelRequested(false); writeLog(`Не удалось отменить анализ: ${anonymizedUiError(error)}`); } };
  const updateItem = (id: string, patch: Partial<PlanItem>) => setResult((current) => current && ({ ...current, items: current.items.map((item) => item.id === id ? { ...item, ...patch } : item) }));
  const apply = async () => { if (!result || !folder) return; setBusy(true); try { const moved = await invoke<number>("apply_sort", { folder, items: result.items.filter((item) => item.included) }); setStatus("Завершено"); writeLog(`Сортировка применена: перемещено ${moved} файлов.`); } catch (error) { setStatus("Ошибка"); writeLog(`Ошибка перемещения: ${anonymizedUiError(error)}`); } finally { setBusy(false); } };
  const undo = async () => { if (!folder) return; setBusy(true); try { const restored = await invoke<number>("undo_last_sort", { folder }); setStatus("Завершено"); writeLog(`Отмена выполнена: восстановлено ${restored} файлов.`); } catch (error) { setStatus("Ошибка"); writeLog(`Ошибка отмены: ${anonymizedUiError(error)}`); } finally { setBusy(false); } };
  const testConnection = async () => { setBusy(true); try { const message = await invoke<string>("test_connection", { ai }); writeLog(message); if (local) await loadModels(); } catch (error) { writeLog(`Не удалось подключиться: ${anonymizedUiError(error)}`); } finally { setBusy(false); } };
  return <main className={dark ? "dark" : ""}><header><h1>AI File Sorter</h1><div className="header-actions"><button className="theme" onClick={toggleTheme} title="Переключить тему">{dark ? "☀ Светлая тема" : "☾ Тёмная тема"}</button><span className={`status status-${status.replaceAll(" ", "-")}`}>{status}</span></div></header>
    <section><h2>Папка</h2><div className="row"><button onClick={chooseFolder} disabled={locked}>Выбрать папку</button><output>{folder || "Папка не выбрана"}</output></div></section>
    <section><h2>Настройки ИИ</h2><div className="grid"><label>Подключение<select value={ai.provider} onChange={e => changeProvider(e.target.value as AiSettings["provider"])} disabled={locked}><option value="lmstudio">LM Studio (локально)</option><option value="ollama">Ollama (локально)</option><option value="compatible">OpenAI-совместимый API</option></select></label><label>Базовый URL<input value={ai.baseUrl} onChange={e => setAi({...ai, baseUrl: e.target.value})} disabled={locked}/></label><label>Модель<input list="available-models" placeholder="Загрузится автоматически или введите вручную" value={ai.model} onChange={e => { userSelectedModel.current = true; setAi({...ai, model: e.target.value}); }} disabled={locked}/><datalist id="available-models">{models.map(model => <option key={model} value={model}/>)}</datalist></label>{!local && <label>API-ключ<input type="password" value={ai.apiKey} onChange={e => setAi({...ai, apiKey: e.target.value})} disabled={locked}/></label>}</div><div className="row"><button onClick={testConnection} disabled={locked}>Проверить подключение</button>{local && <button onClick={() => { userSelectedModel.current = false; void loadModels(); }} disabled={locked}>Использовать активную модель{models.length ? ` (${models.length})` : ""}</button>}{!local && <label className="check"><input type="checkbox" checked={ai.cloudConsent} onChange={e => setAi({...ai, cloudConsent: e.target.checked})} disabled={locked}/> Я понимаю, что имена и извлечённый текст будут отправлены на {ai.baseUrl}</label>}</div></section>
    <section><h2>Режим сортировки</h2><div className="row"><label><input type="radio" checked={sort.mode === "standard"} onChange={() => setSort({...sort, mode: "standard"})} disabled={locked}/> Стандартный</label><label><input type="radio" checked={sort.mode === "custom"} onChange={() => setSort({...sort, mode: "custom"})} disabled={locked}/> Кастомный</label></div>{sort.mode === "custom" && <textarea value={sort.customPrompt} placeholder="Например: отдели бытовые от рабочих и разбей по датам" onChange={e => setSort({...sort, customPrompt: e.target.value})} disabled={locked}/>}<div className="limits"><label className="check"><input type="checkbox" checked={sort.unlimited} onChange={e => setSort({...sort, unlimited: e.target.checked})} disabled={locked}/> Без лимита текста</label><label>Символов на файл<input type="number" min="1" step="1000" disabled={sort.unlimited || locked} value={sort.textLimit} onChange={e => setSort({...sort, textLimit: Math.max(1, Number(e.target.value) || 1)})}/></label><label>Символов всего<input type="number" min="1" step="10000" disabled={sort.unlimited || locked} value={sort.totalLimit} onChange={e => setSort({...sort, totalLimit: Math.max(1, Number(e.target.value) || 1)})}/></label></div><p className="hint">{sort.unlimited ? "Безлимитный режим: весь читаемый текст может быть передан модели. Это заметно увеличивает время анализа и контекст." : `Лимит: до ${sort.textLimit.toLocaleString("ru-RU")} символов на файл, до ${sort.totalLimit.toLocaleString("ru-RU")} символов всего.`}</p><div className="row"><button className="primary" onClick={analyze} disabled={locked}>Анализировать</button>{analyzing && <button className="danger" onClick={cancelAnalysis} disabled={cancelRequested}>{cancelRequested ? "Отмена…" : "Отменить анализ"}</button>}</div>{progress && <div className="analysis-progress" aria-live="polite"><div><strong>{progress.message}</strong><span>Обработано ИИ: {progress.processedFiles}. Ещё не отправлено: {progress.notAttemptedFiles}. На повтор: {progress.retryPendingFiles}.</span></div>{progress.totalBatches > 0 && <progress value={progress.completedBatches} max={progress.totalBatches}/>}</div>}</section>
    <section><h2>Предпросмотр {result && <small>({included} из {result.items.length} будут перемещены)</small>}</h2>{result ? <><p className={`analysis-summary ${result.summary.aiUnprocessed ? "has-errors" : "all-processed"}`}>ИИ обработал: {result.summary.aiProcessed}. Успешно после повтора: {result.summary.retrySucceeded}. В «Не обработано ИИ»: {result.summary.aiUnprocessed}.</p><p className="hint">Проверьте категории и пути. Прокрутите список, чтобы увидеть все файлы.</p><div className="table-wrap"><table><thead><tr><th>Включить</th><th>Исходный путь</th><th>Статус ИИ</th><th>Категория</th><th>Целевой путь</th><th>Уверенность</th><th>Причина</th></tr></thead><tbody>{result.items.map(item => <tr key={item.id} className={item.aiStatus === "unprocessed" ? "ai-unprocessed-row" : undefined}><td><input type="checkbox" checked={item.included} onChange={e => updateItem(item.id, {included: e.target.checked})}/></td><td>{item.relativePath}</td><td><span className={`ai-badge ai-${item.aiStatus}`}>{aiStatusLabels[item.aiStatus]}</span></td><td><input value={item.category} onChange={e => updateItem(item.id, {category: e.target.value})}/></td><td><input value={item.target} onChange={e => updateItem(item.id, {target: e.target.value})}/></td><td>{Math.round(item.confidence * 100)}%</td><td>{item.aiError || item.explanation}{item.warning && <span className="row-warning">{item.warning}</span>}</td></tr>)}</tbody></table></div></> : <p className="hint">После анализа здесь появится план. Ничего не перемещается без подтверждения.</p>}<div className="row"><button className="primary" onClick={apply} disabled={!result || !included || locked}>Применить сортировку</button><button onClick={undo} disabled={!folder || locked}>Отменить последнюю сортировку</button></div></section>
    <section><h2>Обезличенный диагностический лог</h2><p className="hint">Содержит время, проход, номер пакета, длительность, расширения и результат. Имена файлов, пути, содержимое и адрес API не записываются.</p><pre className="diagnostic-log">{log.join("\n")}</pre></section>
  </main>;
}
createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
