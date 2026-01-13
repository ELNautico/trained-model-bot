type Num = number | null | undefined;

function n(row: Record<string, any>, key: string): number | null {
  const v = row?.[key];
  if (v === null || v === undefined) return null;
  const num = Number(v);
  return Number.isFinite(num) ? num : null;
}

// function s(row: Record<string, any>, key: string): string | null {
//   const v = row?.[key];
//   if (v === null || v === undefined) return null;
//   const str = String(v).trim();
//   return str ? str : null;
// }

function fmtFloat(x: Num, digits = 2): string {
  if (x === null || x === undefined) return '';
  return Number(x).toFixed(digits);
}

function pushArg(args: string[], flag: string, value: string | number | null | undefined) {
  if (value === null || value === undefined) return;
  const v = String(value).trim();
  if (!v) return;
  args.push(flag, v);
}

export type BacktestCmdInputs = {
  ticker: string;
  start?: string;
  end?: string;
  metricsStart?: string;
  metricsEnd?: string;
  tag?: string;

  // Optional “portfolio assumptions”
  initialCash?: number;
  risk?: number;
  cooldown?: number;
  noModelExit?: boolean;
};

export function buildBacktestCommand(input: BacktestCmdInputs, row: Record<string, any>) {
  const args: string[] = ['python', 'backtest_signals.py', input.ticker.toUpperCase()];

  pushArg(args, '--start', input.start);
  pushArg(args, '--end', input.end);

  // Parameters: prefer the CSV keys if present
  const entryMinEv = n(row, 'entry_min_ev');
  const retrainEvery = n(row, 'retrain_every');
  const lookback = n(row, 'lookback');

  pushArg(args, '--entry-min-ev', entryMinEv !== null ? fmtFloat(entryMinEv, 2) : null);
  pushArg(args, '--retrain-every', retrainEvery);
  pushArg(args, '--lookback', lookback);

  // If your sweep also includes exit gates, include them if backtest_signals supports them.
  // If it doesn’t, these flags will just be ignored (or error) — so keep these commented
  // until you confirm your CLI supports them.
  // const exitMinEv = n(row, 'exit_min_ev');
  // const exitMinPStop = n(row, 'exit_min_p_stop');
  // pushArg(args, '--exit-min-ev', exitMinEv !== null ? fmtFloat(exitMinEv, 2) : null);
  // pushArg(args, '--exit-min-p-stop', exitMinPStop !== null ? fmtFloat(exitMinPStop, 2) : null);

  pushArg(args, '--metrics-start', input.metricsStart);
  pushArg(args, '--metrics-end', input.metricsEnd);

  // Portfolio assumptions
  if (typeof input.initialCash === 'number') pushArg(args, '--initial-cash', input.initialCash);
  if (typeof input.risk === 'number') pushArg(args, '--risk', input.risk);
  if (typeof input.cooldown === 'number') pushArg(args, '--cooldown', input.cooldown);
  if (input.noModelExit) args.push('--no-model-exit');

  pushArg(args, '--tag', input.tag);

  return args.join(' ');
}

export type NarrowSweepInputs = {
  ticker: string;
  start?: string;
  end?: string;
  metricsStart?: string;
  metricsEnd?: string;
  tag?: string;

  // Narrow sweep controls (editable in UI)
  evSpan?: number;   // +/- span around entry_min_ev
  evStep?: number;   // step size
  retrainEveryChoices?: string; // e.g. "60,80,100"
  lookbackChoices?: string;     // e.g. "1200,1500,2000"
  objective?: string;
  minHoldoutTrades?: number;
  saveTopK?: number;
};

export function buildNarrowSweepCommand(input: NarrowSweepInputs, row: Record<string, any>) {
  const args: string[] = ['python', 'sweep_backtests.py', input.ticker.toUpperCase()];

  pushArg(args, '--start', input.start);
  pushArg(args, '--end', input.end);
  pushArg(args, '--metrics-start', input.metricsStart);
  pushArg(args, '--metrics-end', input.metricsEnd);

  const entryMinEv = n(row, 'entry_min_ev');
  const span = input.evSpan ?? 0.02;
  const step = input.evStep ?? 0.01;

  if (entryMinEv !== null) {
    const a = Math.max(0, entryMinEv - span);
    const b = entryMinEv + span;
    pushArg(args, '--entry-min-ev', `${a.toFixed(2)}:${b.toFixed(2)}:${step.toFixed(2)}`);
  }

  pushArg(args, '--retrain-every', input.retrainEveryChoices ?? undefined);
  pushArg(args, '--lookback', input.lookbackChoices ?? undefined);

  pushArg(args, '--objective', input.objective ?? 'holdout_sharpe');
  if (typeof input.minHoldoutTrades === 'number') pushArg(args, '--min-holdout-trades', input.minHoldoutTrades);
  if (typeof input.saveTopK === 'number') pushArg(args, '--save-top-k', input.saveTopK);

  pushArg(args, '--tag', input.tag);

  return args.join(' ');
}
