import { useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Badge,
  Button,
  Card,
  Divider,
  Group,
  Loader,
  NumberInput,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Text,
  Title,
  Radio,
} from '@mantine/core';
import { IconChartLine, IconDownload, IconInfoCircle } from '@tabler/icons-react';
import MetricsLegend from '../components/MetricsLegend';
import { api, safe } from '../lib/api';
import { fmtInt, fmtNum, fmtPct, fmtDate } from '../lib/format';
import MetricCard from '../components/MetricCard';
import EquityCurveChart from '../components/EquityCurveChart';
import type { RunDetail, RunTopKItem, EquityPoint } from '../types/runs';
import ConfigCommandDrawer from '../components/ConfigCommandDrawer';

type SortKey =
  | 'objective'
  | 'holdout_sharpe'
  | 'holdout_total_return'
  | 'holdout_trades'
  | 'holdout_max_drawdown';

function badgeForStatus(status: unknown) {
  const s = String(status ?? '').toLowerCase();
  if (s === 'ok') return <Badge color="green">ok</Badge>;
  if (s.includes('error')) return <Badge color="red">error</Badge>;
  if (s.includes('filtered')) return <Badge color="yellow">filtered</Badge>;
  return <Badge color="gray">{s || '—'}</Badge>;
}

function numOrNaN(v: unknown): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : Number.NaN;
}

export default function RunDetailPage() {
  const { runId } = useParams();

  const [data, setData] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(true);

  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [minTrades, setMinTrades] = useState<number | undefined>(20);
  const [maxDD, setMaxDD] = useState<number | undefined>(undefined);
  const [sortKey, setSortKey] = useState<SortKey>('objective');
  const [sortDir, setSortDir] = useState<'desc' | 'asc'>('desc');
  const [legendOpen, setLegendOpen] = useState(false);

  const [page, setPage] = useState(1);
  const pageSize = 50;

  // Selection + drawer
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  // Optional Top-K artifacts (only if backend endpoint exists)
  const [topk, setTopk] = useState<RunTopKItem[]>([]);

  // Equity curve for backtest runs
  const [equityCurve, setEquityCurve] = useState<EquityPoint[] | null>(null);

  // Per-rank equity curves for top-K artifacts (rank → data)
  const [topkEquity, setTopkEquity] = useState<Record<number, EquityPoint[]>>({});
  const [topkEquityLoading, setTopkEquityLoading] = useState<Record<number, boolean>>({});

  useEffect(() => {
    if (!runId) return;

    (async () => {
      setLoading(true);

      const d = await safe<RunDetail>(api.getRun(runId), 'Failed to load run');
      setData(d);
      setLoading(false);

      // reset view state on run change
      setPage(1);
      setSelectedIdx(null);
      setDrawerOpen(false);
      setEquityCurve(null);
      setTopkEquity({});
      setTopkEquityLoading({});

      // For backtest runs, auto-fetch equity curve
      if (d?.run?.kind === 'backtest') {
        const eq = await safe<EquityPoint[]>(api.getEquityCurve(runId), 'Failed to load equity curve');
        setEquityCurve(eq ?? null);
      }

      // Try loading top-k; if endpoint isn't present, safe() returns null and we keep empty
      const tk = await safe<RunTopKItem[]>(api.listTopK(runId), 'Failed to load top-k artifacts');
      setTopk(tk ?? []);
    })();
  }, [runId]);

  const run = data?.run;

  const filteredRows = useMemo(() => {
    const rows = data?.rows ?? [];

    return rows.filter((r) => {
      const status = String(r.status ?? '').toLowerCase();

      if (statusFilter !== 'all' && status !== statusFilter) return false;

      if (minTrades != null) {
        const t = numOrNaN(r.holdout_trades);
        if (Number.isFinite(t) && t < minTrades) return false;
      }

      if (maxDD != null) {
        // drawdown is often negative; compare absolute magnitude
        const dd = numOrNaN(r.holdout_max_drawdown);
        if (Number.isFinite(dd) && Math.abs(dd) > maxDD) return false;
      }

      return true;
    });
  }, [data, statusFilter, minTrades, maxDD]);

  const sortedRows = useMemo(() => {
    const rows = [...filteredRows];
    const dir = sortDir === 'desc' ? -1 : 1;

    rows.sort((a, b) => {
      const av = numOrNaN(a[sortKey]);
      const bv = numOrNaN(b[sortKey]);

      const aOK = Number.isFinite(av);
      const bOK = Number.isFinite(bv);

      if (!aOK && !bOK) return 0;
      if (!aOK) return 1;
      if (!bOK) return -1;

      return (av - bv) * dir;
    });

    return rows;
  }, [filteredRows, sortKey, sortDir]);

  const totalPages = Math.max(1, Math.ceil(sortedRows.length / pageSize));

  // Clamp current page if filters reduce totalPages
  useEffect(() => {
    setPage((p) => Math.min(Math.max(1, p), totalPages));
  }, [totalPages]);

  const paged = useMemo(() => {
    const start = (page - 1) * pageSize;
    return sortedRows.slice(start, start + pageSize);
  }, [sortedRows, page]);

  // Selected row on current page
  const selectedRow = useMemo(() => {
    if (selectedIdx === null) return null;
    return paged[selectedIdx] ?? null;
  }, [selectedIdx, paged]);

  // If selection becomes invalid (page/filter changes), clear it
  useEffect(() => {
    if (selectedIdx === null) return;
    if (!paged[selectedIdx]) {
      setSelectedIdx(null);
      setDrawerOpen(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paged]);

  // If filters/sort change, reset selection + page (keeps UX predictable)
  useEffect(() => {
    setPage(1);
    setSelectedIdx(null);
    setDrawerOpen(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [statusFilter, minTrades, maxDD, sortKey, sortDir]);

  const errorSummary = useMemo(() => {
    const rows = data?.rows ?? [];
    const map = new Map<string, number>();
    for (const r of rows) {
      if (String(r.status ?? '').toLowerCase() !== 'error') continue;
      const e = String(r.error ?? 'Unknown error');
      map.set(e, (map.get(e) ?? 0) + 1);
    }
    return [...map.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10);
  }, [data]);

  if (loading) {
    return (
      <Group justify="center" py="xl">
        <Loader />
      </Group>
    );
  }

  if (!data || !run) {
    return <Text c="dimmed">Run not found.</Text>;
  }

  const best = run.best;

  return (
    <>
      <Stack gap="md">
        <Group justify="space-between" align="end">
          <div>
            <Title order={2}>{run.runId}</Title>
            <Text c="dimmed" size="sm">
              {run.ticker} · tag {run.tag ?? '—'} · created {fmtDate(run.createdAt)} · rows {fmtInt(run.rows)}
            </Text>
          </div>

          <Group>
            <Button
              variant="light"
              size="sm"
              leftSection={<IconInfoCircle size={16} />}
              onClick={() => setLegendOpen(true)}
            >
              Legend
            </Button>

            <Button
              component="a"
              href={api.downloadRunCsvUrl(run.runId)}
              leftSection={<IconDownload size={16} />}
              size="sm"
            >
              Download results CSV
            </Button>
          </Group>
        </Group>

        <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }}>
          <MetricCard label="Best holdout Sharpe" value={fmtNum(best?.holdoutSharpe, 2)} />
          <MetricCard label="Best holdout return" value={fmtPct(best?.holdoutReturn)} />
          <MetricCard label="Best holdout drawdown" value={fmtPct(best?.holdoutMaxDrawdown)} />
          <MetricCard label="Best holdout trades" value={fmtInt(best?.holdoutTrades)} />
        </SimpleGrid>

        {/* Equity curve — auto-shown for backtest runs */}
        {equityCurve !== null ? (
          <Card withBorder radius="md" p="md">
            <Group mb="sm">
              <IconChartLine size={18} />
              <Title order={4} style={{ margin: 0 }}>Equity Curve</Title>
              <Text c="dimmed" size="sm">Portfolio value over time · initial ${(100_000).toLocaleString()}</Text>
            </Group>
            <EquityCurveChart data={equityCurve} initialEquity={100_000} height={300} />
          </Card>
        ) : null}

        {topk.length > 0 ? (
          <Card withBorder radius="md" p="md">
            <Group justify="space-between" align="end">
              <div>
                <Title order={4}>Top-K artifacts</Title>
                <Text c="dimmed" size="sm">
                  Detailed reruns saved by <code>--save-top-k</code>.
                </Text>
              </div>
            </Group>

            <Stack gap="md" mt="sm">
              {topk.map((x) => {
                const chartData = topkEquity[x.rank];
                const isLoading = topkEquityLoading[x.rank] ?? false;
                const chartVisible = chartData !== undefined;

                return (
                  <div key={x.rank}>
                    <Group justify="space-between" wrap="nowrap">
                      <Text fw={600}>Rank {x.rank}</Text>
                      <Group gap="xs">
                        {x.hasEquity ? (
                          <Button
                            variant={chartVisible ? 'filled' : 'light'}
                            size="xs"
                            leftSection={<IconChartLine size={14} />}
                            loading={isLoading}
                            onClick={async () => {
                              if (chartVisible) {
                                // toggle off
                                setTopkEquity((prev) => {
                                  const next = { ...prev };
                                  delete next[x.rank];
                                  return next;
                                });
                                return;
                              }
                              setTopkEquityLoading((prev) => ({ ...prev, [x.rank]: true }));
                              const eq = await safe<EquityPoint[]>(
                                api.getTopKEquityCurve(run.runId, x.rank),
                                `Failed to load chart for rank ${x.rank}`,
                              );
                              setTopkEquityLoading((prev) => ({ ...prev, [x.rank]: false }));
                              if (eq) setTopkEquity((prev) => ({ ...prev, [x.rank]: eq }));
                            }}
                          >
                            {chartVisible ? 'Hide chart' : 'View chart'}
                          </Button>
                        ) : null}
                        <Button
                          variant="light"
                          size="xs"
                          leftSection={<IconDownload size={14} />}
                          component="a"
                          href={api.downloadTopKCsvUrl(run.runId, x.rank, 'trades')}
                          disabled={!x.hasTrades}
                        >
                          trades.csv
                        </Button>
                        <Button
                          variant="light"
                          size="xs"
                          leftSection={<IconDownload size={14} />}
                          component="a"
                          href={api.downloadTopKCsvUrl(run.runId, x.rank, 'equity')}
                          disabled={!x.hasEquity}
                        >
                          equity.csv
                        </Button>
                      </Group>
                    </Group>

                    {chartVisible ? (
                      <div style={{ marginTop: 12 }}>
                        <EquityCurveChart data={chartData} initialEquity={100_000} height={240} />
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </Stack>
          </Card>
        ) : null}

        <Card withBorder radius="md" p="md">
          <Group justify="space-between" align="end">
            <Group>
              <Select
                label="Status"
                value={statusFilter}
                onChange={(v) => setStatusFilter(v || 'all')}
                data={[
                  { value: 'all', label: 'All' },
                  { value: 'ok', label: 'ok' },
                  { value: 'filtered_min_trades', label: 'filtered_min_trades' },
                  { value: 'error', label: 'error' },
                ]}
                w={220}
              />

              <NumberInput
                label="Min holdout trades"
                value={minTrades}
                onChange={(v) => setMinTrades(typeof v === 'number' ? v : undefined)}
                min={0}
                w={200}
              />

              <NumberInput
                label="Max |drawdown| (e.g. 0.06 = 6%)"
                value={maxDD}
                onChange={(v) => setMaxDD(typeof v === 'number' ? v : undefined)}
                min={0}
                step={0.01}
                w={260}
              />
            </Group>

            <Group>
              <Select
                label="Sort by"
                value={sortKey}
                onChange={(v) => setSortKey((v as SortKey) || 'objective')}
                data={[
                  { value: 'objective', label: 'objective' },
                  { value: 'holdout_sharpe', label: 'holdout_sharpe' },
                  { value: 'holdout_total_return', label: 'holdout_total_return' },
                  { value: 'holdout_trades', label: 'holdout_trades' },
                  { value: 'holdout_max_drawdown', label: 'holdout_max_drawdown' },
                ]}
                w={220}
              />
              <Select
                label="Direction"
                value={sortDir}
                onChange={(v) => setSortDir((v as 'desc' | 'asc') || 'desc')}
                data={[
                  { value: 'desc', label: 'desc' },
                  { value: 'asc', label: 'asc' },
                ]}
                w={140}
              />
            </Group>
          </Group>

          <Divider my="md" />

          <Text size="sm" c="dimmed" mb="sm">
            Showing {fmtInt(sortedRows.length)} rows (page {page}/{totalPages}, {pageSize}/page)
          </Text>

          <Table highlightOnHover striped withTableBorder>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Select</Table.Th>
                <Table.Th>Status</Table.Th>
                <Table.Th>entry_min_ev</Table.Th>
                <Table.Th>retrain_every</Table.Th>
                <Table.Th>lookback</Table.Th>
                <Table.Th>objective</Table.Th>
                <Table.Th>holdout_sharpe</Table.Th>
                <Table.Th>holdout_return</Table.Th>
                <Table.Th>holdout_dd</Table.Th>
                <Table.Th>holdout_trades</Table.Th>
              </Table.Tr>
            </Table.Thead>

            <Table.Tbody>
              {paged.map((r, idx) => {
                const active = selectedIdx === idx;

                // Stable key: if the row has unique params, use them; otherwise fallback to idx
                const key =
                  `${r.entry_min_ev ?? ''}_${r.retrain_every ?? ''}_${r.lookback ?? ''}_${r.exit_min_ev ?? ''}_${r.exit_min_p_stop ?? ''}_${idx}`;

                return (
                  <Table.Tr
                    key={key}
                    onClick={() => {
                      setSelectedIdx(idx);
                      setDrawerOpen(true);
                    }}
                    style={{
                      cursor: 'pointer',
                      background: active ? 'rgba(34, 139, 230, 0.08)' : undefined,
                    }}
                  >
                    <Table.Td
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedIdx(idx);
                        setDrawerOpen(true);
                      }}
                    >
                      <Radio checked={active} readOnly />
                    </Table.Td>

                    <Table.Td>{badgeForStatus(r.status)}</Table.Td>
                    <Table.Td>{fmtNum(r.entry_min_ev, 2)}</Table.Td>
                    <Table.Td>{fmtInt(r.retrain_every)}</Table.Td>
                    <Table.Td>{fmtInt(r.lookback)}</Table.Td>
                    <Table.Td>{fmtNum(r.objective, 3)}</Table.Td>
                    <Table.Td>{fmtNum(r.holdout_sharpe, 2)}</Table.Td>
                    <Table.Td>{fmtPct(r.holdout_total_return)}</Table.Td>
                    <Table.Td>{fmtPct(r.holdout_max_drawdown)}</Table.Td>
                    <Table.Td>{fmtInt(r.holdout_trades)}</Table.Td>
                  </Table.Tr>
                );
              })}

              {paged.length === 0 ? (
                <Table.Tr>
                  <Table.Td colSpan={10}>
                    <Text c="dimmed" ta="center">
                      No rows match filters.
                    </Text>
                  </Table.Td>
                </Table.Tr>
              ) : null}
            </Table.Tbody>
          </Table>

          <Group justify="space-between" mt="md">
            <Button
              variant="light"
              disabled={page <= 1}
              onClick={() => {
                setPage((p) => Math.max(1, p - 1));
                setSelectedIdx(null);
                setDrawerOpen(false);
              }}
            >
              Prev
            </Button>

            <Group gap="xs">
              <Button
                variant="subtle"
                onClick={() => {
                  setPage(1);
                  setSelectedIdx(null);
                  setDrawerOpen(false);
                }}
              >
                1
              </Button>
              <Text c="dimmed" size="sm">
                …
              </Text>
              <Text size="sm">
                page {page} / {totalPages}
              </Text>
              <Text c="dimmed" size="sm">
                …
              </Text>
              <Button
                variant="subtle"
                onClick={() => {
                  setPage(totalPages);
                  setSelectedIdx(null);
                  setDrawerOpen(false);
                }}
              >
                {totalPages}
              </Button>
            </Group>

            <Button
              variant="light"
              disabled={page >= totalPages}
              onClick={() => {
                setPage((p) => Math.min(totalPages, p + 1));
                setSelectedIdx(null);
                setDrawerOpen(false);
              }}
            >
              Next
            </Button>
          </Group>
        </Card>

        {errorSummary.length > 0 ? (
          <Card withBorder radius="md" p="md">
            <Title order={4}>Top errors (status=error)</Title>
            <Text size="sm" c="dimmed" mb="sm">
              This helps spot systemic issues (bad paths, missing function, etc.).
            </Text>
            <Stack gap="xs">
              {errorSummary.map(([err, count]) => (
                <Group key={err} justify="space-between" align="start">
                  <Text size="sm" style={{ flex: 1 }}>
                    {err}
                  </Text>
                  <Badge color="red">{count}</Badge>
                </Group>
              ))}
            </Stack>
          </Card>
        ) : null}
      </Stack>

      <MetricsLegend opened={legendOpen} onClose={() => setLegendOpen(false)} />

      <ConfigCommandDrawer
        opened={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        ticker={run.ticker}
        row={selectedRow}
      />
    </>
  );
}
