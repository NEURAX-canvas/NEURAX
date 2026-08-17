import React, { createContext, useContext, useState, useCallback, useEffect, useMemo } from 'react';

const API_KEY_STORAGE_KEY = 'neurax_agent_api_key';
const API_PROVIDER_STORAGE_KEY = 'neurax_agent_provider';

export type ApiProvider =
  | 'openai' | 'anthropic' | 'google' | 'mistral'
  | 'fireworks' | 'deepseek' | 'glm' | 'custom';

export interface ApiKeyConfig {
  key: string;
  provider: ApiProvider;
  customEndpoint?: string;
  model?: string;
  label: string;
}

interface ApiKeyContextType {
  config: ApiKeyConfig | null;
  isConfigured: boolean;
  setApiKey: (config: ApiKeyConfig) => void;
  clearApiKey: () => void;
  hasCompletedSetup: boolean;
  markSetupComplete: () => void;
}

const ApiKeyContext = createContext<ApiKeyContextType | undefined>(undefined);

// Kept in sync by hand with neurax-agent/langchain_runner.py's
// OPENAI_COMPATIBLE_DEFAULTS and DEFAULT_*_MODEL constants — the backend is
// what actually talks to each provider, so if the two drift, this panel
// shows a model name the backend won't use.
const PROVIDER_DEFAULTS: Record<ApiProvider, { endpoint: string; defaultModel: string }> = {
  openai: { endpoint: 'https://api.openai.com/v1', defaultModel: 'gpt-4o' },
  anthropic: { endpoint: 'https://api.anthropic.com/v1', defaultModel: 'claude-sonnet-5' },
  google: { endpoint: 'https://generativelanguage.googleapis.com/v1beta', defaultModel: 'gemini-2.5-pro-exp-03-25' },
  mistral: { endpoint: 'https://api.mistral.ai/v1', defaultModel: 'mistral-large-2407' },
  fireworks: { endpoint: 'https://api.fireworks.ai/inference/v1', defaultModel: 'accounts/fireworks/models/llama-v3p1-70b-instruct' },
  deepseek: { endpoint: 'https://api.deepseek.com/v1', defaultModel: 'deepseek-chat' },
  glm: { endpoint: 'https://open.bigmodel.cn/api/paas/v4', defaultModel: 'glm-4-plus' },
  custom: { endpoint: '', defaultModel: 'custom-model' },
};

export function ApiKeyProvider({ children }: { children: React.ReactNode }) {
  const [config, setConfig] = useState<ApiKeyConfig | null>(null);
  const [hasCompletedSetup, setHasCompletedSetup] = useState(false);

  // Restore from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(API_KEY_STORAGE_KEY);
      const storedProvider = localStorage.getItem(API_PROVIDER_STORAGE_KEY);
      if (stored && storedProvider) {
        const parsed = JSON.parse(stored) as ApiKeyConfig;
        setConfig(parsed);
      }
      // Check if setup was completed
      const setupDone = localStorage.getItem('neurax_agent_setup_done') === 'true';
      setHasCompletedSetup(setupDone);
    } catch {
      // Ignore parse errors
    }
  }, []);

  const setApiKey = useCallback((newConfig: ApiKeyConfig) => {
    setConfig(newConfig);
    try {
      localStorage.setItem(API_KEY_STORAGE_KEY, JSON.stringify(newConfig));
      localStorage.setItem(API_PROVIDER_STORAGE_KEY, newConfig.provider);
    } catch {
      // Ignore storage errors
    }
  }, []);

  const clearApiKey = useCallback(() => {
    setConfig(null);
    setHasCompletedSetup(false);
    try {
      localStorage.removeItem(API_KEY_STORAGE_KEY);
      localStorage.removeItem(API_PROVIDER_STORAGE_KEY);
      localStorage.removeItem('neurax_agent_setup_done');
    } catch {
      // Ignore
    }
  }, []);

  const markSetupComplete = useCallback(() => {
    setHasCompletedSetup(true);
    try {
      localStorage.setItem('neurax_agent_setup_done', 'true');
    } catch {
      // Ignore
    }
  }, []);

  const value = useMemo<ApiKeyContextType>(() => ({
    config,
    isConfigured: config !== null && config.key.length > 0,
    setApiKey,
    clearApiKey,
    hasCompletedSetup,
    markSetupComplete,
  }), [config, hasCompletedSetup, setApiKey, clearApiKey, markSetupComplete]);

  return (
    <ApiKeyContext.Provider value={value}>
      {children}
    </ApiKeyContext.Provider>
  );
}

export function useApiKey() {
  const ctx = useContext(ApiKeyContext);
  if (!ctx) throw new Error('useApiKey must be used within ApiKeyProvider');
  return ctx;
}

export { PROVIDER_DEFAULTS };
