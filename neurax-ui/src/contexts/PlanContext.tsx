import React, { createContext, useContext, useMemo } from 'react';
import { PlanTier, PLAN_CONFIGS, PlanConfig } from '@/types/plans.ts';

interface PlanContextType {
  currentPlan: PlanTier;
  planConfig: PlanConfig;
  canAccess: (minPlan: PlanTier) => boolean;
}

const PlanContext = createContext<PlanContextType | undefined>(undefined);

export function PlanProvider({ children }: { children: React.ReactNode }) {
  const currentPlan: PlanTier = 'free';
  const planConfig = PLAN_CONFIGS[currentPlan];

  const canAccess = (_minPlan: PlanTier): boolean => true;

  const value = useMemo(() => ({ currentPlan, planConfig, canAccess }), []);

  return <PlanContext.Provider value={value}>{children}</PlanContext.Provider>;
}

export function usePlan() {
  const context = useContext(PlanContext);
  if (!context) throw new Error('usePlan must be used within PlanProvider');
  return context;
}
