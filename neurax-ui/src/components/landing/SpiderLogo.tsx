import { useRef, useEffect, useCallback } from 'react';

interface Vec3 { x: number; y: number; z: number }
interface Vec2 { x: number; y: number }

const COLORS = {
  body: '#d79921',
  bodyDark: '#b57614',
  leg: '#d65d0e',
  legJoint: '#fe8019',
  web: '#83a598',
  webDim: '#83a59840',
  node: '#98971a',
  nodeGlow: '#b8bb26',
  eye: '#fb4934',
  accent: '#d79921',
};

function project(p: Vec3): Vec2 {
  const fov = 500;
  const factor = fov / (fov + p.z);
  return { x: p.x * factor, y: p.y * factor };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

type SpiderPose = {
  bodyTilt: number;
  legAngles: number[];
  legBend: number[];
};

function idlePose(t: number): SpiderPose {
  const angles: number[] = [];
  const bend: number[] = [];
  for (let i = 0; i < 6; i++) {
    const baseAngle = (i / 6) * Math.PI * 2;
    const phase = i % 2 === 0 ? 0 : Math.PI;
    angles.push(baseAngle + Math.sin(t * 0.5 + phase) * 0.08);
    bend.push(0.6 + Math.sin(t * 0.4 + i * 0.8) * 0.15);
  }
  return {
    bodyTilt: Math.sin(t * 0.3) * 0.03,
    legAngles: angles,
    legBend: bend,
  };
}

export default function SpiderLogo() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const progressRef = useRef(0);
  const targetProgressRef = useRef(1);

  const draw = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, time: number) => {
    ctx.clearRect(0, 0, w, h);

    const progress = progressRef.current;
    const cx = w / 2;
    const cy = h * 0.35;
    const scale = Math.min(w, h) * 0.0015;

    const pose = idlePose(time);

    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(scale, scale);

    const spiderSize = Math.min(w, h) * 0.18;

    ctx.save();
    ctx.rotate(pose.bodyTilt);

    const bodyZ = -30;
    const abdomen = project({ x: 0, y: 18, z: bodyZ + 10 });
    const thorax = project({ x: 0, y: -8, z: bodyZ });
    const head = project({ x: 0, y: -22, z: bodyZ - 5 });

    const bodyGrad = ctx.createRadialGradient(thorax.x, thorax.y, 0, thorax.x, thorax.y, 30 * spiderSize / 100);
    bodyGrad.addColorStop(0, COLORS.body);
    bodyGrad.addColorStop(0.6, COLORS.bodyDark);
    bodyGrad.addColorStop(1, '#1d2021');
    ctx.fillStyle = bodyGrad;

    ctx.beginPath();
    ctx.ellipse(thorax.x, thorax.y, 20 * spiderSize / 100, 14 * spiderSize / 100, 0, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.ellipse(abdomen.x, abdomen.y, 24 * spiderSize / 100, 18 * spiderSize / 100, 0.1, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.ellipse(head.x, head.y, 10 * spiderSize / 100, 8 * spiderSize / 100, 0, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.bodyDark;
    ctx.fill();

    for (let i = 0; i < 6; i++) {
      const angle = pose.legAngles[i];
      const bend = pose.legBend[i];
      const isRight = i < 3;
      const side = isRight ? 1 : -1;
      const legIndex = isRight ? i : i - 3;

      const baseX = thorax.x + Math.cos(angle) * 8 * spiderSize / 100;
      const baseY = thorax.y + Math.sin(angle) * 8 * spiderSize / 100;

      const seg1Len = 50 * spiderSize / 100;
      const seg2Len = 45 * spiderSize / 100;

      const a1 = angle + side * (0.3 + legIndex * 0.08) + Math.sin(time * 0.5 + i) * 0.03;
      const j1x = baseX + Math.cos(a1) * seg1Len;
      const j1y = baseY + Math.sin(a1) * seg1Len;

      const a2 = a1 + side * (0.8 - bend * 0.3) + Math.sin(time * 0.6 + i * 1.2) * 0.02;
      const tipX = j1x + Math.cos(a2) * seg2Len;
      const tipY = j1y + Math.sin(a2) * seg2Len;

      ctx.beginPath();
      ctx.moveTo(baseX, baseY);
      ctx.lineTo(j1x, j1y);
      ctx.lineTo(tipX, tipY);
      ctx.strokeStyle = COLORS.leg;
      ctx.lineWidth = 2.5 * spiderSize / 100;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(j1x, j1y, 2.5 * spiderSize / 100, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.legJoint;
      ctx.fill();
    }

    const eyeR = project({ x: -5, y: -23, z: bodyZ - 8 });
    const eyeL = project({ x: 5, y: -23, z: bodyZ - 8 });
    const eyeSize = 2.5 * spiderSize / 100;

    ctx.beginPath();
    ctx.arc(eyeR.x, eyeR.y, eyeSize, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.eye;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(eyeL.x, eyeL.y, eyeSize, 0, Math.PI * 2);
    ctx.fill();

    const eyeGlowR = project({ x: -5, y: -24, z: bodyZ - 10 });
    const eyeGlowL = project({ x: 5, y: -24, z: bodyZ - 10 });

    const glowSize = 8 * spiderSize / 100;
    const eyeGlow1 = ctx.createRadialGradient(eyeGlowR.x, eyeGlowR.y, 0, eyeGlowR.x, eyeGlowR.y, glowSize);
    eyeGlow1.addColorStop(0, `${COLORS.eye}60`);
    eyeGlow1.addColorStop(1, 'transparent');
    ctx.fillStyle = eyeGlow1;
    ctx.fillRect(eyeGlowR.x - glowSize, eyeGlowR.y - glowSize, glowSize * 2, glowSize * 2);

    const eyeGlow2 = ctx.createRadialGradient(eyeGlowL.x, eyeGlowL.y, 0, eyeGlowL.x, eyeGlowL.y, glowSize);
    eyeGlow2.addColorStop(0, `${COLORS.eye}60`);
    eyeGlow2.addColorStop(1, 'transparent');
    ctx.fillStyle = eyeGlow2;
    ctx.fillRect(eyeGlowL.x - glowSize, eyeGlowL.y - glowSize, glowSize * 2, glowSize * 2);

    ctx.restore();

    const webProgress = clamp(progress, 0, 1);
    const webLength = h * 0.55 * webProgress;
    const webStartY = 25 * spiderSize / 100;
    const webEndY = webStartY + webLength;
    const webMaxWidth = Math.min(w * 0.5, 300);

    if (webProgress > 0) {
      for (let strand = 0; strand < 7; strand++) {
        const strandAngle = (strand / 6 - 0.5) * Math.PI * 0.6;

        ctx.beginPath();
        ctx.moveTo(0, webStartY);

        const cpX = Math.sin(strandAngle) * webMaxWidth * 0.5;
        const cpY = webStartY + webLength * 0.3;
        const endX = Math.sin(strandAngle) * webMaxWidth * 0.85;
        const endY = webEndY;

        ctx.quadraticCurveTo(cpX, cpY, endX, endY);
        ctx.strokeStyle = strand === 3 ? COLORS.web : COLORS.webDim;
        ctx.lineWidth = strand === 3 ? 1.5 * spiderSize / 100 : 0.8 * spiderSize / 100;
        ctx.stroke();
      }

      const crossLines = 8;
      const webProgressRatio = webLength / (h * 0.55);
      const visibleCrossLines = Math.floor(crossLines * webProgressRatio);

      for (let c = 0; c < visibleCrossLines; c++) {
        const t = (c + 1) / crossLines;
        const y = webStartY + webLength * t;
        const spreadAtY = webMaxWidth * 0.85 * t;

        ctx.beginPath();
        ctx.ellipse(0, y, spreadAtY, 4 * spiderSize / 100, 0, 0, Math.PI * 2);
        ctx.strokeStyle = COLORS.webDim;
        ctx.lineWidth = 0.5 * spiderSize / 100;
        ctx.stroke();
      }

      const numNodes = 15;
      const visibleNodes = Math.floor(numNodes * webProgressRatio);

      for (let n = 0; n < visibleNodes; n++) {
        const t = (n + 1) / numNodes;
        const y = webStartY + webLength * t;
        const spreadAtY = webMaxWidth * 0.85 * t;
        const nx = Math.sin(n * 2.5 + time) * spreadAtY * 0.5;
        const ny = y + Math.cos(n * 1.7 + time * 0.5) * 8 * spiderSize / 100;

        const pulse = Math.sin(time * 2 + n * 1.3) * 0.3 + 0.7;
        const nodeSize = (2 + Math.sin(n * 0.9) * 1.5) * spiderSize / 100;

        const grad = ctx.createRadialGradient(nx, ny, 0, nx, ny, nodeSize * 3);
        grad.addColorStop(0, `${COLORS.nodeGlow}${Math.floor(80 * pulse).toString(16).padStart(2, '0')}`);
        grad.addColorStop(0.5, `${COLORS.node}${Math.floor(60 * pulse).toString(16).padStart(2, '0')}`);
        grad.addColorStop(1, 'transparent');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(nx, ny, nodeSize * 3, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.arc(nx, ny, nodeSize * 0.7, 0, Math.PI * 2);
        ctx.fillStyle = COLORS.nodeGlow;
        ctx.fill();

        if (n < visibleNodes - 1) {
          const t2 = (n + 2) / numNodes;
          const y2 = webStartY + webLength * t2;
          const spreadAtY2 = webMaxWidth * 0.85 * t2;
          const nx2 = Math.sin((n + 1) * 2.5 + time) * spreadAtY2 * 0.5;
          const ny2 = y2 + Math.cos((n + 1) * 1.7 + time * 0.5) * 8 * spiderSize / 100;

          if (Math.random() > 0.3 || n % 3 === 0) {
            ctx.beginPath();
            ctx.moveTo(nx, ny);
            ctx.lineTo(nx2, ny2);
            ctx.strokeStyle = `${COLORS.web}${Math.floor(30 * pulse).toString(16).padStart(2, '0')}`;
            ctx.lineWidth = 0.3 * spiderSize / 100;
            ctx.stroke();
          }
        }
      }
    }

    ctx.restore();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let running = true;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    let startTime = performance.now();

    const animate = (now: number) => {
      if (!running) return;
      const elapsed = (now - startTime) / 1000;

      const currentTarget = targetProgressRef.current;
      progressRef.current = lerp(progressRef.current, currentTarget, 0.02);
      if (Math.abs(progressRef.current - currentTarget) < 0.001) {
        progressRef.current = currentTarget;
      }

      draw(ctx, canvas.width, canvas.height, elapsed);
      animRef.current = requestAnimationFrame(animate);
    };

    animRef.current = requestAnimationFrame(animate);

    const handleScroll = () => {
      const scrollY = window.scrollY;
      const maxScroll = window.innerHeight * 0.5;
      const p = clamp(scrollY / maxScroll, 0, 1);
      targetProgressRef.current = p;

      const opacity = 0.4 - p * 0.25;
      canvas.style.opacity = String(Math.max(opacity, 0.1));
    };

    window.addEventListener('scroll', handleScroll, { passive: true });

    return () => {
      running = false;
      cancelAnimationFrame(animRef.current);
      window.removeEventListener('resize', resize);
      window.removeEventListener('scroll', handleScroll);
    };
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none"
      style={{ zIndex: 0, opacity: 0.35, transition: 'opacity 0.3s' }}
    />
  );
}
