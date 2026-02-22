import { useMemo } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  type TooltipProps,
} from 'recharts';
import { Text } from '@mantine/core';
import type { EquityPoint } from '../types/runs';

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmtCurrency(v: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(v);
}

/** Abbreviate ISO date string to "MMM DD 'YY" for X-axis ticks. */
function fmtTick(dateStr: string): string {
  try {
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
  } catch {
    return dateStr;
  }
}

/** Full date for tooltip. */
function fmtFull(dateStr: string): string {
  try {
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
  } catch {
    return dateStr;
  }
}

// ── Custom tooltip ────────────────────────────────────────────────────────────

function CustomTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null;

  const point = payload[0]?.payload as EquityPoint | undefined;
  if (!point) return null;

  const equity = payload[0]?.value ?? 0;
  const action = point.action;

  const actionColor =
    action === 'BUY' ? '#40c057' : action === 'SELL' ? '#fa5252' : undefined;

  return (
    <div
      style={{
        background: 'rgba(30,30,40,0.95)',
        border: '1px solid rgba(255,255,255,0.12)',
        borderRadius: 6,
        padding: '8px 12px',
        fontSize: 13,
        lineHeight: 1.6,
      }}
    >
      <div style={{ color: '#aaa', marginBottom: 2 }}>{fmtFull(point.date)}</div>
      <div style={{ color: '#fff', fontWeight: 600 }}>{fmtCurrency(Number(equity))}</div>
      {action && action !== 'HOLD' && action !== 'WAIT' && (
        <div style={{ color: actionColor ?? '#ccc', fontSize: 11, marginTop: 2 }}>
          {action}
        </div>
      )}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

interface Props {
  data: EquityPoint[];
  initialEquity?: number;
  height?: number;
}

export default function EquityCurveChart({ data, initialEquity = 100_000, height = 280 }: Props) {
  if (!data.length) {
    return (
      <Text c="dimmed" ta="center" py="md" size="sm">
        No equity data available.
      </Text>
    );
  }

  // Thin out X-axis ticks so they don't overlap (show ~8 labels max).
  const tickIndices = useMemo(() => {
    const n = data.length;
    if (n <= 8) return data.map((_, i) => i);
    const step = Math.ceil(n / 8);
    const indices: number[] = [];
    for (let i = 0; i < n; i += step) indices.push(i);
    if (indices[indices.length - 1] !== n - 1) indices.push(n - 1);
    return indices;
  }, [data]);

  const ticks = tickIndices.map((i) => data[i].date);

  const minEquity = Math.min(...data.map((d) => d.equity));
  const maxEquity = Math.max(...data.map((d) => d.equity));
  const padding = (maxEquity - minEquity) * 0.05 || initialEquity * 0.05;
  const yMin = Math.floor((minEquity - padding) / 1000) * 1000;
  const yMax = Math.ceil((maxEquity + padding) / 1000) * 1000;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 16 }}>
        <defs>
          <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#228be6" stopOpacity={0.25} />
            <stop offset="95%" stopColor="#228be6" stopOpacity={0.02} />
          </linearGradient>
        </defs>

        <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />

        <XAxis
          dataKey="date"
          ticks={ticks}
          tickFormatter={fmtTick}
          tick={{ fill: '#868e96', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />

        <YAxis
          domain={[yMin, yMax]}
          tickFormatter={fmtCurrency}
          tick={{ fill: '#868e96', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={80}
        />

        <Tooltip content={<CustomTooltip />} />

        {/* Reference line at initial equity */}
        <ReferenceLine
          y={initialEquity}
          stroke="rgba(255,255,255,0.18)"
          strokeDasharray="4 3"
        />

        <Area
          type="monotone"
          dataKey="equity"
          stroke="#228be6"
          strokeWidth={2}
          fill="url(#equityGradient)"
          dot={false}
          activeDot={{ r: 4, fill: '#228be6', stroke: '#fff', strokeWidth: 1.5 }}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
