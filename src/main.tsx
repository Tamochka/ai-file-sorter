import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { open } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import type { AiSettings, AnalysisResult, PlanItem, SortSettings, Status } from "./types";
import "./styles.css";

const initialAi: AiSettings = { provider: "lmstudio", baseUrl: "http://127.0.0.1:1234/v1", model: "", apiKey: "", cloudConsent: false };
const initialSort: SortSettings = { mode: "standard", customPrompt: "", textLimit: 12000, totalLimit: 500000 };
type ModelList = { models: string[]; activeModel?: string };

function App() {
  const [folder, setFolder] = useState(""); const [ai, setAi] = useState(initialAi); const [sort, setSort] = useState(initialSort);
  const [status, setStatus] = useState<Status>("Готово"); const [log, setLog] = useState<string[]>(["Выберите папку для начала работы."]);
  const [result, setResult] = useState<AnalysisResult | null>(null); const [busy, setBusy] = useState(false); const [dark, setDark] = useState(() => localStorage.getItem("theme") === "dark");
  const [models, setModels] = useState<string[]>([]);
  const userSelectedModel = useRef(false);
  useEffect(() => { document.documentElement.dataset.theme = dark ? "dark" : "light"; }, [dark]);
  const included = useMemo(() => result?.items.filter((item) => item.included).length ?? 0, [result]);
  const local = ai.provider === "lmstudio" || ai.provider === "ollama" || /^(https?:\/\/)?(127\.0\.0\.1|localhost)/.test(ai.baseUrl);
  const writeLog = (message: string) => setLog((current) => [...current, message]);
  const loadModels = async (settings = ai) => { try { const found = await invoke<ModelList>("list_models", { ai: settings }); setModels(found.models); setAi(current => current.provider === settings.provider && current.baseUrl === settings.baseUrl && !userSelectedModel.current ? { ...current, model: found.activeModel || found.models[0] } : current); writeLog(found.activeModel ? `Активная модель LM Studio: ${found.activeModel}` : `Найдены модели: ${found.models.length}.`); } catch (error) { setModels([]); writeLog(`Не удалось получить список моделей: ${String(error)}`); } };
  useEffect(() => { if (ai.provider === "lmstudio" || ai.provider === "ollama") void loadModels(); }, [ai.provider]);
  const changeProvider = (provider: AiSettings["provider"]) => { const baseUrl = provider === "ollama" ? "http://127.0.0.1:11434" : provider === "lmstudio" ? "http://127.0.0.1:1234/v1" : ai.baseUrl; userSelectedModel.current = false; setModels([]); setAi({ ...ai, provider, baseUrl, model: "" }); };
  const chooseFolder = async () => { try { const selected = await open({ directory: true, multiple: false, title: "Выберите папку" }); if (typeof selected === "string") { setFolder(selected); writeLog(`Выбрана папка: ${selected}`); } else { writeLog("Выбор папки отменён."); } } catch (error) { setStatus("Ошибка"); writeLog(`Не удалось открыть диалог выбора папки: ${String(error)}`); } };
  const toggleTheme = () => { const next = !dark; setDark(next); localStorage.setItem("theme", next ? "dark" : "light"); };
  const analyze = async () => {
    if (!folder) return writeLog("Ошибка: сначала выберите папку.");
    if (!ai.model.trim()) return writeLog("Ошибка: укажите имя модели.");
    if (!local && !ai.cloudConsent) return writeLog("Нужно подтвердить передачу данных в облачный сервис.");
    setBusy(true); setStatus("Идёт анализ"); writeLog("Сканирование файлов и подготовка безопасного плана…");
    try { const data = await invoke<AnalysisResult>("analyze_folder", { folder, ai, sort }); setResult(data); setStatus("Требуется подтверждение"); writeLog(`План готов: ${data.totalFiles} файлов, приблизительно ${data.estimatedChars.toLocaleString("ru-RU")} символов.`); data.warnings.forEach(writeLog); }
    catch (error) { setStatus("Ошибка"); writeLog(`Ошибка анализа: ${String(error)}`); } finally { setBusy(false); }
  };
  const updateItem = (id: string, patch: Partial<PlanItem>) => setResult((current) => current && ({ ...current, items: current.items.map((item) => item.id === id ? { ...item, ...patch } : item) }));
  const apply = async () => { if (!result || !folder) return; setBusy(true); try { const moved = await invoke<number>("apply_sort", { folder, items: result.items.filter((item) => item.included) }); setStatus("Завершено"); writeLog(`Сортировка применена: перемещено ${moved} файлов.`); } catch (error) { setStatus("Ошибка"); writeLog(`Ошибка перемещения: ${String(error)}`); } finally { setBusy(false); } };
  const undo = async () => { if (!folder) return; setBusy(true); try { const restored = await invoke<number>("undo_last_sort", { folder }); setStatus("Завершено"); writeLog(`Отмена выполнена: восстановлено ${restored} файлов.`); } catch (error) { setStatus("Ошибка"); writeLog(`Ошибка отмены: ${String(error)}`); } finally { setBusy(false); } };
  const testConnection = async () => { setBusy(true); try { const message = await invoke<string>("test_connection", { ai }); writeLog(message); if (local) await loadModels(); } catch (error) { writeLog(`Не удалось подключиться: ${String(error)}`); } finally { setBusy(false); } };
  return <main className={dark ? "dark" : ""}><header><h1>AI File Sorter</h1><div className="header-actions"><button className="theme" onClick={toggleTheme} title="Переключить тему">{dark ? "☀ Светлая тема" : "☾ Тёмная тема"}</button><span className={`status status-${status.replaceAll(" ", "-")}`}>{status}</span></div></header>
    <section><h2>Папка</h2><div className="row"><button onClick={chooseFolder} disabled={busy}>Выбрать папку</button><output>{folder || "Папка не выбрана"}</output></div></section>
    <section><h2>Настройки ИИ</h2><div className="grid"><label>Подключение<select value={ai.provider} onChange={e => changeProvider(e.target.value as AiSettings["provider"])}><option value="lmstudio">LM Studio (локально)</option><option value="ollama">Ollama (локально)</option><option value="compatible">OpenAI-совместимый API</option></select></label><label>Базовый URL<input value={ai.baseUrl} onChange={e => setAi({...ai, baseUrl: e.target.value})}/></label><label>Модель<input list="available-models" placeholder="Загрузится автоматически или введите вручную" value={ai.model} onChange={e => { userSelectedModel.current = true; setAi({...ai, model: e.target.value}); }}/><datalist id="available-models">{models.map(model => <option key={model} value={model}/>)}</datalist></label>{!local && <label>API-ключ<input type="password" value={ai.apiKey} onChange={e => setAi({...ai, apiKey: e.target.value})}/></label>}</div><div className="row"><button onClick={testConnection} disabled={busy}>Проверить подключение</button>{local && <button onClick={() => { userSelectedModel.current = false; void loadModels(); }} disabled={busy}>Использовать активную модель{models.length ? ` (${models.length})` : ""}</button>}{!local && <label className="check"><input type="checkbox" checked={ai.cloudConsent} onChange={e => setAi({...ai, cloudConsent: e.target.checked})}/> Я понимаю, что имена и извлечённый текст будут отправлены на {ai.baseUrl}</label>}</div></section>
    <section><h2>Режим сортировки</h2><div className="row"><label><input type="radio" checked={sort.mode === "standard"} onChange={() => setSort({...sort, mode: "standard"})}/> Стандартный</label><label><input type="radio" checked={sort.mode === "custom"} onChange={() => setSort({...sort, mode: "custom"})}/> Кастомный</label></div>{sort.mode === "custom" && <textarea value={sort.customPrompt} placeholder="Например: отдели бытовые от рабочих и разбей по датам" onChange={e => setSort({...sort, customPrompt: e.target.value})}/>}<p className="hint">Лимит: до {sort.textLimit.toLocaleString("ru-RU")} символов на файл, до {sort.totalLimit.toLocaleString("ru-RU")} символов всего.</p><button className="primary" onClick={analyze} disabled={busy}>Анализировать</button></section>
    <section><h2>Предпросмотр {result && <small>({included} из {result.items.length} будут перемещены)</small>}</h2>{result ? <><p className="hint">Проверьте категории и пути. Прокрутите список, чтобы увидеть все файлы.</p><div className="table-wrap"><table><thead><tr><th>Включить</th><th>Исходный путь</th><th>Категория</th><th>Целевой путь</th><th>Уверенность</th><th>Причина</th></tr></thead><tbody>{result.items.map(item => <tr key={item.id}><td><input type="checkbox" checked={item.included} onChange={e => updateItem(item.id, {included: e.target.checked})}/></td><td>{item.relativePath}</td><td><input value={item.category} onChange={e => updateItem(item.id, {category: e.target.value})}/></td><td><input value={item.target} onChange={e => updateItem(item.id, {target: e.target.value})}/></td><td>{Math.round(item.confidence * 100)}%</td><td>{item.warning || item.explanation}</td></tr>)}</tbody></table></div></> : <p className="hint">После анализа здесь появится план. Ничего не перемещается без подтверждения.</p>}<div className="row"><button className="primary" onClick={apply} disabled={!result || !included || busy}>Применить сортировку</button><button onClick={undo} disabled={!folder || busy}>Отменить последнюю сортировку</button></div></section>
    <section><h2>Лог</h2><pre>{log.join("\n")}</pre></section>
  </main>;
}
createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
