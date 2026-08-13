export type SortMode = "standard" | "custom";
export type ProviderKind = "lmstudio" | "ollama" | "compatible";
export type Status = "Готово" | "Идёт анализ" | "Требуется подтверждение" | "Ошибка" | "Завершено";

export interface AiSettings { provider: ProviderKind; baseUrl: string; model: string; apiKey: string; cloudConsent: boolean; }
export interface SortSettings { mode: SortMode; customPrompt: string; textLimit: number; totalLimit: number; unlimited: boolean; }
export interface PlanItem { id: string; source: string; relativePath: string; target: string; category: string; explanation: string; confidence: number; included: boolean; warning?: string; }
export interface AnalysisResult { items: PlanItem[]; totalFiles: number; estimatedChars: number; warnings: string[]; }
