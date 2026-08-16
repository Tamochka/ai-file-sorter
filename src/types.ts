export type SortMode = "standard" | "custom";
export type ProviderKind = "lmstudio" | "ollama" | "compatible";
export type Status = "Готово" | "Идёт анализ" | "Требуется подтверждение" | "Ошибка" | "Завершено";
export type AiStatus = "processed" | "retry_pending" | "unprocessed";

export interface AiSettings { provider: ProviderKind; baseUrl: string; model: string; apiKey: string; cloudConsent: boolean; }
export interface SortSettings { mode: SortMode; customPrompt: string; textLimit: number; totalLimit: number; unlimited: boolean; }
export interface PlanItem { id: string; source: string; relativePath: string; target: string; category: string; explanation: string; confidence: number; included: boolean; warning?: string; aiStatus: AiStatus; aiError?: string; }
export interface AiSummary { aiProcessed: number; retrySucceeded: number; aiUnprocessed: number; }
export interface AnalysisResult { items: PlanItem[]; totalFiles: number; estimatedChars: number; warnings: string[]; summary: AiSummary; }
export interface AnalysisProgress { phase: "scanning" | "main" | "retry" | "complete"; completedBatches: number; totalBatches: number; processedFiles: number; pendingFiles: number; notAttemptedFiles: number; retryPendingFiles: number; message: string; }
export interface ExtensionCount { extension: string; count: number; }
export interface AnalysisLogEvent { phase: "scanning" | "main" | "retry"; attempt?: number; batchNumber?: number; totalBatches?: number; fileCount: number; extensions: ExtensionCount[]; durationMs: number; outcome: "success" | "partial" | "error" | "cancelled"; successfulFiles: number; unresolvedFiles: number; skippedFiles: number; inputBytes?: number; errorKind?: string; errorDetail?: string; }
