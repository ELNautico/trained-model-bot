import { useMemo, useState } from 'react';
import {
  Button,
  Code,
  Drawer,
  Group,
  NumberInput,
  Stack,
  Text,
  TextInput,
  Title,
  CopyButton,
  Tooltip,
  Divider,
} from '@mantine/core';
import { IconCheck, IconCopy } from '@tabler/icons-react';
import { buildBacktestCommand, buildNarrowSweepCommand } from '../lib/commands';

type Props = {
  opened: boolean;
  onClose: () => void;
  ticker: string;
  row: Record<string, any> | null;
};

export default function ConfigCommandDrawer({ opened, onClose, ticker, row }: Props) {
  const [start, setStart] = useState('2018-01-01');
  const [end, setEnd] = useState('2025-12-31');
  const [metricsStart, setMetricsStart] = useState('2024-01-01');
  const [metricsEnd, setMetricsEnd] = useState('2025-12-31');
  const [tag, setTag] = useState('FINAL_from_UI');

  const [initialCash, setInitialCash] = useState<number>(100000);
  const [risk, setRisk] = useState<number>(0.01);
  const [cooldown, setCooldown] = useState<number>(0);

  // Narrow sweep knobs
  const [evSpan, setEvSpan] = useState<number>(0.02);
  const [evStep, setEvStep] = useState<number>(0.01);
  const [retrainChoices, setRetrainChoices] = useState<string>('60,80,100');
  const [lookbackChoices, setLookbackChoices] = useState<string>('1200,1500,2000');
  const [minHoldoutTrades, setMinHoldoutTrades] = useState<number>(20);
  const [saveTopK, setSaveTopK] = useState<number>(3);

  const backtestCmd = useMemo(() => {
    if (!row) return '';
    return buildBacktestCommand(
      { ticker, start, end, metricsStart, metricsEnd, tag, initialCash, risk, cooldown },
      row
    );
  }, [row, ticker, start, end, metricsStart, metricsEnd, tag, initialCash, risk, cooldown]);

  const narrowSweepCmd = useMemo(() => {
    if (!row) return '';
    return buildNarrowSweepCommand(
      {
        ticker,
        start,
        end,
        metricsStart,
        metricsEnd,
        tag: `${tag}_narrow`,
        evSpan,
        evStep,
        retrainEveryChoices: retrainChoices,
        lookbackChoices,
        objective: 'holdout_sharpe',
        minHoldoutTrades,
        saveTopK,
      },
      row
    );
  }, [row, ticker, start, end, metricsStart, metricsEnd, tag, evSpan, evStep, retrainChoices, lookbackChoices, minHoldoutTrades, saveTopK]);

  return (
    <Drawer opened={opened} onClose={onClose} position="right" size="lg" title="Selected config">
      {!row ? (
        <Text c="dimmed">Select a row to generate commands.</Text>
      ) : (
        <Stack gap="md">
          <Title order={4}>Run inputs</Title>
          <Group grow>
            <TextInput label="Start" value={start} onChange={(e) => setStart(e.currentTarget.value)} />
            <TextInput label="End" value={end} onChange={(e) => setEnd(e.currentTarget.value)} />
          </Group>
          <Group grow>
            <TextInput label="Holdout start" value={metricsStart} onChange={(e) => setMetricsStart(e.currentTarget.value)} />
            <TextInput label="Holdout end" value={metricsEnd} onChange={(e) => setMetricsEnd(e.currentTarget.value)} />
          </Group>
          <TextInput label="Tag" value={tag} onChange={(e) => setTag(e.currentTarget.value)} />

          <Divider />

          <Title order={4}>Portfolio assumptions (optional)</Title>
          <Group grow>
            <NumberInput label="Initial cash" value={initialCash} onChange={(v) => setInitialCash(Number(v))} min={0} />
            <NumberInput label="Risk" value={risk} onChange={(v) => setRisk(Number(v))} min={0} step={0.001} decimalScale={3} />
            <NumberInput label="Cooldown" value={cooldown} onChange={(v) => setCooldown(Number(v))} min={0} />
          </Group>

          <Divider />

          <Title order={4}>Backtest command</Title>
          <Code block style={{ whiteSpace: 'pre-wrap' }}>
            {backtestCmd}
          </Code>
          <Group justify="end">
            <CopyButton value={backtestCmd}>
              {({ copied, copy }) => (
                <Tooltip label={copied ? 'Copied' : 'Copy command'}>
                  <Button onClick={copy} leftSection={copied ? <IconCheck size={16} /> : <IconCopy size={16} />}>
                    Copy
                  </Button>
                </Tooltip>
              )}
            </CopyButton>
          </Group>

          <Divider />

          <Title order={4}>Optional: narrow sweep around entry_min_ev</Title>
          <Text c="dimmed" size="sm">
            This is a convenience command to refine around the selected config. Adjust ranges as needed.
          </Text>

          <Group grow>
            <NumberInput label="EV span (±)" value={evSpan} onChange={(v) => setEvSpan(Number(v))} min={0} step={0.01} decimalScale={2} />
            <NumberInput label="EV step" value={evStep} onChange={(v) => setEvStep(Number(v))} min={0.001} step={0.01} decimalScale={2} />
          </Group>
          <Group grow>
            <TextInput label="retrain choices" value={retrainChoices} onChange={(e) => setRetrainChoices(e.currentTarget.value)} />
            <TextInput label="lookback choices" value={lookbackChoices} onChange={(e) => setLookbackChoices(e.currentTarget.value)} />
          </Group>
          <Group grow>
            <NumberInput label="Min holdout trades" value={minHoldoutTrades} onChange={(v) => setMinHoldoutTrades(Number(v))} min={0} />
            <NumberInput label="Save top-k" value={saveTopK} onChange={(v) => setSaveTopK(Number(v))} min={0} />
          </Group>

          <Code block style={{ whiteSpace: 'pre-wrap' }}>
            {narrowSweepCmd}
          </Code>
          <Group justify="end">
            <CopyButton value={narrowSweepCmd}>
              {({ copied, copy }) => (
                <Tooltip label={copied ? 'Copied' : 'Copy command'}>
                  <Button variant="light" onClick={copy} leftSection={copied ? <IconCheck size={16} /> : <IconCopy size={16} />}>
                    Copy
                  </Button>
                </Tooltip>
              )}
            </CopyButton>
          </Group>
        </Stack>
      )}
    </Drawer>
  );
}
