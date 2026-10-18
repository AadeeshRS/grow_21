export const DEFAULTS = { scaffold:{Lmax:3,failuresPerScaffoldLevel:2,inactivityMs:10000,startLevelDefault:0}, integration:{engineTimeoutMs:3000,telemetryAckTimeoutMs:1500,maxFallbackTasks:2,queueMaxEvents:500}, telemetry:{maxPathHz:60,maxPathPoints:600,maxSubEvents:100}, attemptFailure:{resumeTimeoutMs:3000,consecutiveMissLimit:4,dropFailLimit:3,incorrectFlipLimit:3} } as const;
export const clamp=(n:number,min=0,max=1)=>Math.max(min,Math.min(max,n));
export const lerp=(a:number,b:number,d:number)=>a+(b-a)*clamp(d);
export const roundLerp=(a:number,b:number,d:number)=>Math.round(lerp(a,b,d));
