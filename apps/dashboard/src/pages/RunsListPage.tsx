import { useEffect, useMemo, useState } from 'react';
import {
  Badge,
  Button,
  Card,
  Group,
  Loader,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
} from '@mantine/core';
import { IconDownload, IconSearch } from '@tabler/icons-react';
import { Link } from 'react-router-dom';
import { api, safe } from '../lib/api';
import { fmtDate, fmtInt, fmtNum, fmtPct } from '../lib/format';
import type { RunSummary } from '../types/runs';

function statusBadge(status?: string | null) {
  const s = (status ?? '').toLowerCase();
  if (s === 'ok') return <Badge color="green">ok</Badge>;
  if (s.includes('error')) return <Badge color="red">error</Badge>;
  return <Badge color="gray">{status ?? '—'}</Badge>;
}

export default function RunsListPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [q, setQ] = useState('');

  useEffect(() => {
    (async () => {
      setLoading(true);
      const data = await safe<RunSummary[]>(api.listRuns(), 'Failed to load runs');
      setRuns(data ?? []);
      setLoading(false);
    })();
  }, []);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return runs;
    return runs.filter((r) => {
      const hay = `${r.ticker} ${r.tag ?? ''} ${r.runId}`.toLowerCase();
      return hay.includes(s);
    });
  }, [runs, q]);

  return (
    <Stack gap="md">
      <Group justify="space-between" align="end">
        <div>
          <Title order={2}>Sweep Runs</Title>
          <Text c="dimmed" size="sm">
            Reads results from <code>backtests/*_sweep_results*.csv</code>
          </Text>
        </div>

        <TextInput
          value={q}
          onChange={(e) => setQ(e.currentTarget.value)}
          leftSection={<IconSearch size={16} />}
          placeholder="Search ticker, tag, runId…"
          w={320}
        />
      </Group>

      <Card withBorder radius="md" p="md">
        {loading ? (
          <Group justify="center" py="lg">
            <Loader />
          </Group>
        ) : (
          <Table highlightOnHover striped>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Run</Table.Th>
                <Table.Th>Ticker</Table.Th>
                <Table.Th>Tag</Table.Th>
                <Table.Th>Created</Table.Th>
                <Table.Th>Rows</Table.Th>
                <Table.Th>Best (holdout)</Table.Th>
                <Table.Th />
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {filtered.map((r) => {
                const b = r.best;
                return (
                  <Table.Tr key={r.runId}>
                    <Table.Td>
                      <Text component={Link} to={`/runs/${encodeURIComponent(r.runId)}`} fw={600}>
                        {r.runId}
                      </Text>
                    </Table.Td>
                    <Table.Td>{r.ticker}</Table.Td>
                    <Table.Td>{r.tag ?? '—'}</Table.Td>
                    <Table.Td>{fmtDate(r.createdAt)}</Table.Td>
                    <Table.Td>{fmtInt(r.rows)}</Table.Td>
                    <Table.Td>
                      <Group gap="xs">
                        {statusBadge(b?.status ?? null)}
                        <Text size="sm">
                          Sharpe {fmtNum(b?.holdoutSharpe, 2)} · Ret {fmtPct(b?.holdoutReturn)} · DD{' '}
                          {fmtPct(b?.holdoutMaxDrawdown)} · Trades {fmtInt(b?.holdoutTrades)}
                        </Text>
                      </Group>
                      {b?.entryMinEv != null ? (
                        <Text size="xs" c="dimmed" mt={4}>
                          entry_ev={fmtNum(b.entryMinEv, 2)} · retrain={fmtInt(b.retrainEvery)} · lookback={fmtInt(b.lookback)}
                        </Text>
                      ) : null}
                    </Table.Td>
                    <Table.Td>
                      <Group justify="end">
                        <Button
                          component="a"
                          href={api.downloadRunCsvUrl(r.runId)}
                          variant="light"
                          leftSection={<IconDownload size={16} />}
                        >
                          CSV
                        </Button>
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                );
              })}
              {filtered.length === 0 ? (
                <Table.Tr>
                  <Table.Td colSpan={7}>
                    <Text c="dimmed" ta="center">
                      No runs found.
                    </Text>
                  </Table.Td>
                </Table.Tr>
              ) : null}
            </Table.Tbody>
          </Table>
        )}
      </Card>
    </Stack>
  );
}
