import { describe, it, expect } from 'vitest';
import { getParamsForFamily, getRequiredParamsForFamily, getOptionalParamsForFamily } from '@/utils/hyperparameterDefs';
import { ArchitectureFamily } from '@/types/plugins';

const FAMILIES: ArchitectureFamily[] = ['transformer','moe','cnn','diffusion','ssm','gnn','rnn','gan','snn','rl','experimental'];

describe('hyperparameter coverage per family', () => {
  it('reports totals', () => {
    for (const f of FAMILIES) {
      const all = getParamsForFamily(f);
      const req = getRequiredParamsForFamily(f);
      const opt = getOptionalParamsForFamily(f);
      console.log(`${f.padEnd(14)} total=${String(all.length).padStart(3)}  requis=${String(req.length).padStart(2)}  facultatifs=${String(opt.length).padStart(3)}`);
      expect(all.length).toBe(req.length + opt.length);
    }
  });

  it('every family exposes the universal training controls', () => {
    for (const f of FAMILIES) {
      const keys = new Set(getParamsForFamily(f).map(p => String(p.key)));
      for (const k of ['learningRate','batchSize','optimizer','warmupSteps','weightDecay',
                       'tensorParallel','pipelineParallel','gradientCheckpointing','zeroStage']) {
        expect(keys.has(k), `${f} should expose ${k}`).toBe(true);
      }
    }
  });

  it('required params are marked and non-empty for real families', () => {
    for (const f of FAMILIES) {
      const req = getRequiredParamsForFamily(f).map(p => String(p.key));
      expect(req).toContain('batchSize');
      for (const p of getRequiredParamsForFamily(f)) expect(p.required).toBe(true);
    }
  });

  it('no duplicate keys within a family', () => {
    for (const f of FAMILIES) {
      const keys = getParamsForFamily(f).map(p => String(p.key));
      const dupes = keys.filter((k,i) => keys.indexOf(k) !== i);
      expect(dupes, `${f} has duplicate keys: ${dupes.join(',')}`).toEqual([]);
    }
  });

  it('every param has label, description, type and default', () => {
    for (const f of FAMILIES) {
      for (const p of getParamsForFamily(f)) {
        expect(p.label, `${f}.${String(p.key)} label`).toBeTruthy();
        expect(p.description.length, `${f}.${String(p.key)} description`).toBeGreaterThan(10);
        expect(['int','float','categorical','bool']).toContain(p.type);
        expect(p.defaultValue, `${f}.${String(p.key)} default`).not.toBeUndefined();
      }
    }
  });
});
