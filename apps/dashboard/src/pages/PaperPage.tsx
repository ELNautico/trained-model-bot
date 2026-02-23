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
} from 'recharts';
import MetricCard from '../components/MetricCard';
import EquityCurveChart from '../components/EquityCurveChart';
import { api, safe } from '../lib/api';
import type { PaperSummary, PaperPosition, PaperTrade, PaperEquityPoint } from '../types/paper';
import type { EquityPoint } from '../types/runs';

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

function fmtPct(v: number | null | undefined): string {
  if (v == null) return '—';
  return (v * 100).toFixed(1) + '%';
}

/** For values already in percent (3.5 = +3.5%) */
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

const PIE_COLORS = ['#228be6', '#40c057', '#fab005', '#fa5252', '#7950f2', '#fd7e14'];

// ── Sub-components ────────────────────────────────────────────────────────────

function ExitReasonsPie({ data }: { data: Record<string, number> }) {
  const entries = Object.entries(data);
  if (entries.length === 0) {
    return (
      <Center h={200}>
        <Text c="dimmed" size="sm">
          No exit data yet.
        </Text>
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
        <Text c="dimmed" size="sm">
          No trades yet.
        </Text>
      </Center>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={trades} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
        <XAxis dataKey="ticker" tick={{ fill: '#868e96', fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis
          tickFormatter={(v) => fmtUsd(v)}
          tick={{ fill: '#868e96', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={72}
        />
        <Tooltip formatter={(v: number) => [fmtUsd(v), 'P&L']} />
        <Bar dataKey="pnl" radius={[3, 3, 0, 0]}>
          {trades.map((t, i) => (
            <Cell key={i} fill={(t.pnl ?? 0) >= 0 ? '#40c057' : '#fa5252'} />
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
        </Table.Tr>
      </Table.Thead>
      <Table.Tbody>
        {positions.map((p) => (
          <Table.Tr key={p.id}>
            <Table.Td>
              <Badge variant="light">{p.ticker}</Badge>
            </Table.Td>
            <Table.Td>{fmtDate(p.entry_ts)}</Table.Td>
            <Table.Td>{fmtUsd(p.entry_px)}</Table.Td>
            <Table.Td>{p.shares}</Table.Td>
            <Table.Td>{fmtUsd(p.stop_px)}</Table.Td>
            <Table.Td>{fmtUsd(p.target_px)}</Table.Td>
          </Table.Tr>
        ))}
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
            <Table.Td style={{ color: (t.pnl ?? 0) >= 0 ? '#40c057' : '#fa5252', fontWeight: 600 }}>
              {fmtUsd(t.pnl)}
            </Table.Td>
            <Table.Td style={{ color: (t.return_pct ?? 0) >= 0 ? '#40c057' : '#fa5252' }}>
              {fmtReturnPct(t.return_pct)}
            </Table.Td>
            <Table.Td>
              <Badge size="xs" variant="outline" color="gray">
                {t.exit_reason}
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

  return (
    <Stack gap="lg">
      <Group justify="space-between" align="flex-end">
        <div>
          <Title order={2}>Paper Trading</Title>
          <Text c="dimmed" size="sm">
            Simulated dry-run performance — fake money, real signals
          </Text>
        </div>
        {loading && (
          <Text c="dimmed" size="sm">
            Loading…
          </Text>
        )}
      </Group>

      {/* Row 1: primary stats */}
      <SimpleGrid cols={{ base: 2, sm: 4 }}>
        <MetricCard
          label="Total P&L"
          value={fmtUsd(summary?.totalPnl ?? 0)}
          sub={`Starting equity: $100,000`}
        />
        <MetricCard
          label="Win Rate"
          value={fmtPct(summary?.winRate ?? 0)}
          sub={`${summary?.totalTrades ?? 0} closed trades`}
        />
        <MetricCard
          label="Total Trades"
          value={String(summary?.totalTrades ?? 0)}
        />
        <MetricCard
          label="Avg Return / Trade"
          value={fmtReturnPct(summary?.avgReturnPct ?? 0)}
        />
      </SimpleGrid>

      {/* Row 2: secondary stats */}
      <SimpleGrid cols={{ base: 2, sm: 4 }}>
        <MetricCard
          label="Best Trade"
          value={fmtUsd(summary?.bestTradePnl ?? null)}
        />
        <MetricCard
          label="Worst Trade"
          value={fmtUsd(summary?.worstTradePnl ?? null)}
        />
        <MetricCard
          label="Avg Hold Days"
          value={fmtDays(summary?.avgHoldDays ?? null)}
        />
        <MetricCard
          label="Open Positions"
          value={String(summary?.openPositions ?? 0)}
        />
      </SimpleGrid>

      {/* Equity curve */}
      <Card withBorder radius="md" p="md">
        <Text fw={600} mb="sm">
          Equity Curve
        </Text>
        <EquityCurveChart
          data={equity as unknown as EquityPoint[]}
          initialEquity={100_000}
          height={280}
        />
      </Card>

      {/* Charts row */}
      <Grid>
        <Grid.Col span={{ base: 12, sm: 6 }}>
          <Card withBorder radius="md" p="md" h="100%">
            <Text fw={600} mb="sm">
              Exit Reasons
            </Text>
            <ExitReasonsPie data={summary?.exitReasons ?? {}} />
          </Card>
        </Grid.Col>
        <Grid.Col span={{ base: 12, sm: 6 }}>
          <Card withBorder radius="md" p="md" h="100%">
            <Text fw={600} mb="sm">
              P&L Per Trade
            </Text>
            <PnlBarChart trades={trades} />
          </Card>
        </Grid.Col>
      </Grid>

      {/* Open positions */}
      <Card withBorder radius="md" p="md">
        <Text fw={600} mb="sm">
          Open Positions ({positions.length})
        </Text>
        <PositionsTable positions={positions} />
      </Card>

      {/* Trade history */}
      <Card withBorder radius="md" p="md">
        <Text fw={600} mb="sm">
          Trade History ({trades.length})
        </Text>
        <TradesTable trades={trades} />
      </Card>
    </Stack>
  );
}
