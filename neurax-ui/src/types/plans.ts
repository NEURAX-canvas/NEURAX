export type PlanTier = 'free';

export interface PlanConfig {
  id: PlanTier;
  name: string;
  displayName: string;
  color: string;
  badge: string;
  features: string[];
}

export const PLAN_CONFIGS: Record<PlanTier, PlanConfig> = {
  free: {
    id: 'free',
    name: 'Neurax (OSS)',
    displayName: 'OSS',
    color: 'hsl(239, 84%, 67%)',
    badge: 'bg-primary/10 text-primary border-primary/30',
    features: [
      'Everything — open source',
    ],
  },
};

export function canAccessArchitecture(_plan: PlanTier, _architectureId: string): boolean {
  return true;
}

export function getRequiredPlanName(_minPlan: PlanTier): string {
  return 'Neurax (OSS)';
}
