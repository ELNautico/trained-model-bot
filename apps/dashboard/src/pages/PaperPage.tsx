import { useEffect, useState } from 'react';
import {
  Title,
  Text,
  SimpleGrid,
  Card,
  Group,
  Badge,
  Table,
  Stack,
  Grid,
  Center,
} from '@mantine/core';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  Cell,
  PieChart,
  Pie,
  Tooltip,
  ResponsiveContainer,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';
import MetricCard from '../components/MetricCard';
import EquityCurveChart from '../components/EquityCurveChart';
import { api, safe } from '../lib/api';
import type { PaperSummary, PaperPosition, PaperTrade, PaperEquityPoint, TickerStat } from '../types/paper';
import type { EquityPoint } from '../types/runs';

// ── Colours ───────────────────────────────────────────────────────────────────
const GREEN = '#40c057';
const YELLOW = '#fab005';
const RED = '#fa5252';
const DIM = '#868e96';
const PIE_COLORS = ['#228be6', GREEN, YELLOW, RED, '#7950f2', '#fd7e14'];

function pnlColor(v: number): string {
  if (v > 0) return GREEN;
  if (v < 0) return RED;
  return DIM;
}

function winRateColor(v: number): string {
  if (v >= 0.55) return GREEN;
  if (v >= 0.42) return YELLOW;
  return RED;
}

function sharpeColor(v: number | null): string | undefined {
  if (v == null) return undefined;
  if (v >= 1.0) return GREEN;
  if (v >= 0.5) return YELLOW;
  return RED;
}

function pfColor(v: number | null, totalTrades: number): string | undefined {
  if (totalTrades === 0) return undefined;
  if (v === null) return GREEN; // no losses → infinite PF
  if (v >= 1.5) return GREEN;
  if (v >= 1.0) return YELLOW;
  return RED;
}

function ddColor(v: number): string {
  if (v <= 5) return GREEN;
  if (v <= 15) return YELLOW;
  return RED;
}

function cagrColor(v: number | null): string | undefined {
  if (v == null) return undefined;
  if (v > 0.1) return GREEN;
  if (v >= 0) return YELLOW;
  return RED;
}

// ── Formatters ────────────────────────────────────────────────────────────────
function fmtUsd(v: number | null | undefined): string {
  if (v == null) return '—';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(v);
}

function fmtUsdSigned(v: number | null | undefined): string {
  if (v == null) return '—';
  const formatted = fmtUsd(v);
  return v > 0 ? '+' + formatted : formatted;
}

function fmtPct(v: number | null | undefined): string {
  if (v == null) return '—';
  return (v * 100).toFixed(1) + '%';
}

function fmtReturnPct(v: number | null | undefined): string {
  if (v == null) return '—';
  const sign = v >= 0 ? '+' : '';
  return `${sign}${v.toFixed(2)}%`;
}

function fmtDays(v: number | null | undefined): string {
  if (v == null) return '—';
  return v.toFixed(1) + 'd';
}

function fmtDate(ts: string | undefined): string {
  if (!ts) return '—';
  try {
    return new Date(ts).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return ts;
  }
}

function fmtSharpe(v: number | null): string {
  if (v == null) return '—';
  return v.toFixed(2);
}

function fmtProfitFactor(v: number | null, totalTrades: number): string {
  if (totalTrades === 0) return '—';
  if (v === null) return '∞'; // no losses
  return v.toFixed(2) + '×';
}

function fmtDrawdown(v: number): string {
  if (v === 0) return '—';
  return `−${v.toFixed(1)}%`;
}

function fmtCagr(v: number | null): string {
  if (v === null) return '—';
  const sign = v >= 0 ? '+' : '';
  return `${sign}${(v * 100).toFixed(1)}% p.a.`;
}

// ── Sub-components ────────────────────────────────────────────────────────────

function ExitReasonsPie({ data }: { data: Record<string, number> }) {
  const entries = Object.entries(data);
  if (entries.length === 0) {
    return (
      <Center h={200}>
        <Text c="dimmed" size="sm">No exit data yet.</Text>
      </Center>
    );
  }

  const chartData = entries.map(([name, value]) => ({ name, value }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <PieChart>
        <Pie
          data={chartData}
          dataKey="value"
          nameKey="name"
          cx="50%"
          cy="50%"
          outerRadius={80}
          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
          labelLine={false}
        >
          {chartData.map((_, i) => (
            <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
          ))}
        </Pie>
        <Tooltip formatter={(v: number) => [v, 'trades']} />
      </PieChart>
    </ResponsiveContainer>
  );
}

function PnlBarChart({ trades }: { trades: PaperTrade[] }) {
  if (trades.length === 0) {
    return (
      <Center h={200}>
        <Text c="dimmed" size="sm">No trades yet.</Text>
      </Center>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={trades} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
        <XAxis dataKey="ticker" tick={{ fill: DIM, fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis
          tickFormatter={(v) => fmtUsd(v)}
          tick={{ fill: DIM, fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={72}
        />
        <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
        <Tooltip formatter={(v: number) => [fmtUsd(v), 'P&L']} />
        <Bar dataKey="pnl" radius={[3, 3, 0, 0]}>
          {trades.map((t, i) => (
            <Cell key={i} fill={(t.pnl ?? 0) >= 0 ? GREEN : RED} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function DrawdownChart({ data }: { data: PaperEquityPoint[] }) {
  const nonZero = data.filter((d) => d.drawdown < 0);
  if (data.length < 2 || nonZero.length === 0) {
    return (
      <Center h={120}>
        <Text c="dimmed" size="sm">No drawdown recorded yet.</Text>
      </Center>
    );
  }

  // Thin the X-axis labels to avoid crowding
  const interval = data.length <= 10 ? 0 : Math.floor(data.length / 8);

  return (
    <ResponsiveContainer width="100%" height={140}>
      <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <defs>
          <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={RED} stopOpacity={0.5} />
            <stop offset="95%" stopColor={RED} stopOpacity={0.05} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
        <XAxis
          dataKey="date"
          tick={{ fill: DIM, fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          interval={interval}
        />
        <YAxis
          domain={['dataMin', 0]}
          tickFormatter={(v: number) => `${v.toFixed(0)}%`}
          tick={{ fill: DIM, fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={44}
        />
        <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
        <Tooltip formatter={(v: number) => [`${v.toFixed(2)}%`, 'Drawdown']} />
        <Area
          type="monotone"
          dataKey="drawdown"
          stroke={RED}
          strokeWidth={1.5}
          fill="url(#ddGrad)"
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function TickerBreakdownChart({ tickerStats }: { tickerStats: TickerStat[] }) {
  if (tickerStats.length === 0) {
    return (
      <Center h={180}>
        <Text c="dimmed" size="sm">No ticker data yet.</Text>
      </Center>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={tickerStats} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
        <XAxis dataKey="ticker" tick={{ fill: DIM, fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis
          tickFormatter={(v) => fmtUsd(v)}
          tick={{ fill: DIM, fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={72}
        />
        <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
        <Tooltip
          formatter={(_v: number, _name: string, entry) => {
            const stat = entry.payload as TickerStat;
            return [
              `${fmtUsd(stat.totalPnl)} · WR ${(stat.winRate * 100).toFixed(0)}% · ${stat.trades} trades`,
              stat.ticker,
            ];
          }}
        />
        <Bar dataKey="totalPnl" name="P&L" radius={[3, 3, 0, 0]}>
          {tickerStats.map((s, i) => (
            <Cell key={i} fill={s.totalPnl >= 0 ? GREEN : RED} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function PositionsTable({ positions }: { positions: PaperPosition[] }) {
  if (positions.length === 0) {
    return (
      <Text c="dimmed" size="sm" ta="center" py="md">
        No open positions.
      </Text>
    );
  }

  function holdColor(hold: number, horizon: number): string {
    const pct = horizon > 0 ? hold / horizon : 0;
    if (pct >= 0.8) return 'red';
    if (pct >= 0.5) return 'yellow';
    return 'green';
  }

  return (
    <Table striped highlightOnHover withTableBorder withColumnBorders>
      <Table.Thead>
        <Table.Tr>
          <Table.Th>Ticker</Table.Th>
          <Table.Th>Entry Date</Table.Th>
          <Table.Th>Entry Px</Table.Th>
          <Table.Th>Shares</Table.Th>
          <Table.Th>Stop</Table.Th>
          <Table.Th>Target</Table.Th>
          <Table.Th>Risk to Stop</Table.Th>
          <Table.Th>Upside</Table.Th>
          <Table.Th>Hold</Table.Th>
        </Table.Tr>
      </Table.Thead>
      <Table.Tbody>
        {positions.map((p) => {
          const riskPct = p.entry_px > 0
            ? ((p.entry_px - p.stop_px) / p.entry_px) * 100
            : 0;
          const upsidePct = p.entry_px > 0
            ? ((p.target_px - p.entry_px) / p.entry_px) * 100
            : 0;
          const hold = p.hold_days ?? 0;
          const horizon = p.horizon_days ?? 10;
          return (
            <Table.Tr key={p.id}>
              <Table.Td>
                <Badge variant="light">{p.ticker}</Badge>
              </Table.Td>
              <Table.Td>{fmtDate(p.entry_ts)}</Table.Td>
              <Table.Td>{fmtUsd(p.entry_px)}</Table.Td>
              <Table.Td>{p.shares}</Table.Td>
              <Table.Td style={{ color: RED }}>{fmtUsd(p.stop_px)}</Table.Td>
              <Table.Td style={{ color: GREEN }}>{fmtUsd(p.target_px)}</Table.Td>
              <Table.Td style={{ color: RED }}>−{riskPct.toFixed(1)}%</Table.Td>
              <Table.Td style={{ color: GREEN }}>+{upsidePct.toFixed(1)}%</Table.Td>
              <Table.Td>
                <Badge size="sm" color={holdColor(hold, horizon)} variant="light">
                  {hold}/{horizon}d
                </Badge>
              </Table.Td>
            </Table.Tr>
          );
        })}
      </Table.Tbody>
    </Table>
  );
}

function TradesTable({ trades }: { trades: PaperTrade[] }) {
  if (trades.length === 0) {
    return (
      <Text c="dimmed" size="sm" ta="center" py="md">
        No closed trades yet.
      </Text>
    );
  }

  const reasonColor = (r: string) => {
    if (r === 'take_profit') return 'green';
    if (r === 'stop_loss') return 'red';
    return 'gray';
  };

  return (
    <Table striped highlightOnHover withTableBorder withColumnBorders>
      <Table.Thead>
        <Table.Tr>
          <Table.Th>Ticker</Table.Th>
          <Table.Th>Entry</Table.Th>
          <Table.Th>Exit</Table.Th>
          <Table.Th>Entry Px</Table.Th>
          <Table.Th>Exit Px</Table.Th>
          <Table.Th>Shares</Table.Th>
          <Table.Th>P&L</Table.Th>
          <Table.Th>Return</Table.Th>
          <Table.Th>Reason</Table.Th>
        </Table.Tr>
      </Table.Thead>
      <Table.Tbody>
        {[...trades].reverse().map((t) => (
          <Table.Tr key={t.id}>
            <Table.Td>
              <Badge variant="light">{t.ticker}</Badge>
            </Table.Td>
            <Table.Td>{fmtDate(t.entry_ts)}</Table.Td>
            <Table.Td>{fmtDate(t.exit_ts)}</Table.Td>
            <Table.Td>{fmtUsd(t.entry_px)}</Table.Td>
            <Table.Td>{fmtUsd(t.exit_px)}</Table.Td>
            <Table.Td>{t.shares}</Table.Td>
            <Table.Td style={{ color: (t.pnl ?? 0) >= 0 ? GREEN : RED, fontWeight: 600 }}>
              {fmtUsdSigned(t.pnl)}
            </Table.Td>
            <Table.Td style={{ color: (t.return_pct ?? 0) >= 0 ? GREEN : RED }}>
              {fmtReturnPct(t.return_pct)}
            </Table.Td>
            <Table.Td>
              <Badge size="xs" color={reasonColor(t.exit_reason ?? '')} variant="light">
                {t.exit_reason ?? '—'}
              </Badge>
            </Table.Td>
          </Table.Tr>
        ))}
      </Table.Tbody>
    </Table>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function PaperPage() {
  const [summary, setSummary] = useState<PaperSummary | null>(null);
  const [positions, setPositions] = useState<PaperPosition[]>([]);
  const [trades, setTrades] = useState<PaperTrade[]>([]);
  const [equity, setEquity] = useState<PaperEquityPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    void (async () => {
      setLoading(true);
      const [s, pos, tr, eq] = await Promise.all([
        safe(api.getPaperSummary(), 'Paper summary failed'),
        safe(api.getPaperPositions(), 'Paper positions failed'),
        safe(api.getPaperTrades(), 'Paper trades failed'),
        safe(api.getPaperEquity(), 'Paper equity failed'),
      ]);
      if (s) setSummary(s);
      if (pos) setPositions(pos);
      if (tr) setTrades(tr);
      if (eq) setEquity(eq);
      setLoading(false);
    })();
  }, []);

  const n = summary?.totalTrades ?? 0;
  const finalEquity = 100_000 + (summary?.totalPnl ?? 0);

  return (
    <Stack gap="lg">
      {/* ── Header ── */}
      <Group justify="space-between" align="flex-end">
        <div>
          <Title order={2}>Paper Trading</Title>
          <Text c="dimmed" size="sm">
            Simulated dry-run — fake money, real signals · Starting equity: $100,000
          </Text>
        </div>
        {loading && <Text c="dimmed" size="sm">Loading…</Text>}
      </Group>

      {/* ── Row 1: Performance ── */}
      <SimpleGrid cols={{ base: 2, sm: 4 }}>
        <MetricCard
          label="Total P&L"
          value={fmtUsdSigned(summary?.totalPnl ?? 0)}
          sub={`Current equity: ${fmtUsd(finalEquity)}`}
          valueColor={pnlColor(summary?.totalPnl ?? 0)}
        />
        <MetricCard
          label="CAGR"
          value={fmtCagr(summary?.cagr ?? null)}
          sub="Annualised return"
          valueColor={cagrColor(summary?.cagr ?? null)}
        />
        <MetricCard
          label="Sharpe Ratio"
          value={fmtSharpe(summary?.sharpeRatio ?? null)}
          sub="Trade-level, annualised"
          valueColor={sharpeColor(summary?.sharpeRatio ?? null)}
        />
        <MetricCard
          label="Profit Factor"
          value={fmtProfitFactor(summary?.profitFactor ?? null, n)}
          sub="Gross wins / gross losses"
          valueColor={pfColor(summary?.profitFactor ?? null, n)}
        />
      </SimpleGrid>

      {/* ── Row 2: Risk ── */}
      <SimpleGrid cols={{ base: 2, sm: 4 }}>
        <MetricCard
          label="Max Drawdown"
          value={fmtDrawdown(summary?.maxDrawdownPct ?? 0)}
          sub="Peak-to-trough"
          valueColor={summary?.maxDrawdownPct ? ddColor(summary.maxDrawdownPct) : undefined}
        />
        <MetricCard
          label="Win Rate"
          value={fmtPct(summary?.winRate ?? 0)}
          sub={`${summary?.totalTrades ?? 0} closed trades`}
          valueColor={winRateColor(summary?.winRate ?? 0)}
        />
        <MetricCard
          label="Expectancy / Trade"
          value={fmtUsdSigned(summary?.expectancy ?? 0)}
          sub="Avg $ earned per trade"
          valueColor={pnlColor(summary?.expectancy ?? 0)}
        />
        <MetricCard
          label="Avg Return / Trade"
          value={fmtReturnPct(summary?.avgReturnPct ?? 0)}
          valueColor={pnlColor(summary?.avgReturnPct ?? 0)}
        />
      </SimpleGrid>

      {/* ── Row 3: Trade stats ── */}
      <SimpleGrid cols={{ base: 2, sm: 4 }}>
        <MetricCard
          label="Total Trades"
          value={String(n)}
          sub={`Avg hold: ${fmtDays(summary?.avgHoldDays ?? null)}`}
        />
        <MetricCard
          label="Open Positions"
          value={String(summary?.openPositions ?? 0)}
        />
        <MetricCard
          label="Best Trade"
          value={fmtUsdSigned(summary?.bestTradePnl ?? null)}
          valueColor={summary?.bestTradePnl != null ? GREEN : undefined}
        />
        <MetricCard
          label="Worst Trade"
          value={fmtUsdSigned(summary?.worstTradePnl ?? null)}
          sub={`Max consec. losses: ${summary?.maxConsecLosses ?? 0}`}
          valueColor={summary?.worstTradePnl != null ? RED : undefined}
        />
      </SimpleGrid>

      {/* ── Equity Curve ── */}
      <Card withBorder radius="md" p="md">
        <Text fw={600} mb="sm">Equity Curve</Text>
        <EquityCurveChart
          data={equity as unknown as EquityPoint[]}
          initialEquity={100_000}
          height={280}
        />
      </Card>

      {/* ── Drawdown ── */}
      <Card withBorder radius="md" p="md">
        <Group justify="space-between" mb="sm">
          <Text fw={600}>Drawdown</Text>
          {summary && summary.maxDrawdownPct > 0 && (
            <Badge
              color={
                ddColor(summary.maxDrawdownPct) === GREEN
                  ? 'green'
                  : ddColor(summary.maxDrawdownPct) === YELLOW
                    ? 'yellow'
                    : 'red'
              }
              variant="light"
            >
              Max −{summary.maxDrawdownPct.toFixed(1)}%
            </Badge>
          )}
        </Group>
        <DrawdownChart data={equity} />
      </Card>

      {/* ── Charts row ── */}
      <Grid>
        <Grid.Col span={{ base: 12, sm: 6 }}>
          <Card withBorder radius="md" p="md" h="100%">
            <Text fw={600} mb="sm">Exit Reasons</Text>
            <ExitReasonsPie data={summary?.exitReasons ?? {}} />
          </Card>
        </Grid.Col>
        <Grid.Col span={{ base: 12, sm: 6 }}>
          <Card withBorder radius="md" p="md" h="100%">
            <Text fw={600} mb="sm">P&L by Ticker</Text>
            <TickerBreakdownChart tickerStats={summary?.tickerStats ?? []} />
          </Card>
        </Grid.Col>
      </Grid>

      {/* ── P&L per trade ── */}
      <Card withBorder radius="md" p="md">
        <Text fw={600} mb="sm">P&L Per Trade (chronological)</Text>
        <PnlBarChart trades={trades} />
      </Card>

      {/* ── Open positions ── */}
      <Card withBorder radius="md" p="md">
        <Text fw={600} mb="sm">Open Positions ({positions.length})</Text>
        <PositionsTable positions={positions} />
      </Card>

      {/* ── Trade history ── */}
      <Card withBorder radius="md" p="md">
        <Text fw={600} mb="sm">Trade History ({trades.length})</Text>
        <TradesTable trades={trades} />
      </Card>
    </Stack>
  );
}
