import { Badge, Card, Group, Modal, SimpleGrid, Stack, Text, Title } from '@mantine/core';

type Props = {
  opened: boolean;
  onClose: () => void;
};

function MetricItem(props: {
  name: string;
  meaning: string;
  why: string;
  tips?: string[];
  badge?: { label: string; color?: string };
}) {
  return (
    <Card withBorder radius="md" p="md">
      <Group justify="space-between" align="start" mb="xs">
        <Title order={5}>{props.name}</Title>
        {props.badge ? <Badge color={props.badge.color ?? 'gray'}>{props.badge.label}</Badge> : null}
      </Group>

      <Stack gap={6}>
        <Text size="sm">
          <Text span fw={600}>
            What it is:
          </Text>{' '}
          {props.meaning}
        </Text>

        <Text size="sm">
          <Text span fw={600}>
            Why it matters:
          </Text>{' '}
          {props.why}
        </Text>

        {props.tips?.length ? (
          <Stack gap={4} mt={4}>
            <Text size="sm" fw={600}>
              How to read it:
            </Text>
            {props.tips.map((t) => (
              <Text key={t} size="sm" c="dimmed">
                • {t}
              </Text>
            ))}
          </Stack>
        ) : null}
      </Stack>
    </Card>
  );
}

export default function MetricsLegend({ opened, onClose }: Props) {
  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title={<Title order={4}>Metrics legend</Title>}
      size="xl"
      centered
    >
      <Stack gap="md">
        <Text c="dimmed" size="sm">
          These metrics summarize a backtest run (equity curve + trades). “Holdout” means results
          measured only inside your evaluation window (e.g. 2024–2025), which is the safest way to
          select configs.
        </Text>

        <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="md">
          <MetricItem
            name="objective"
            badge={{ label: 'ranking', color: 'blue' }}
            meaning="The single score used to rank configs in this sweep (often holdout_sharpe)."
            why="It tells you which row the sweep considers “best” under your chosen objective."
            tips={[
              'Higher is better.',
              'Treat as a ranking tool, not a guarantee of real-world performance.',
              'Always sanity-check trades and drawdown.',
            ]}
          />

          <MetricItem
            name="holdout_sharpe"
            badge={{ label: 'risk-adjusted', color: 'violet' }}
            meaning="Return per unit of equity volatility (daily Sharpe of the holdout equity curve)."
            why="Helps prefer smoother performance instead of raw return spikes."
            tips={[
              'Higher is better.',
              '~0 = noise; ~1 = solid; >1.5 can be strong but verify sample size.',
              'Very high Sharpe with few trades is often overfitting.',
            ]}
          />

          <MetricItem
            name="holdout_total_return"
            badge={{ label: 'profit', color: 'green' }}
            meaning="Percent change in equity from the start to the end of the holdout window."
            why="It answers the basic question: did the strategy make money during the period you care about?"
            tips={[
              'Higher is better.',
              'Compare alongside drawdown to understand risk.',
              'A big return with big drawdown may be unacceptable operationally.',
            ]}
          />

          <MetricItem
            name="holdout_max_drawdown"
            badge={{ label: 'risk', color: 'red' }}
            meaning="Worst peak-to-trough drop in the equity curve during holdout (often negative)."
            why="Drawdown is the most practical pain metric: it measures how bad it gets before it recovers."
            tips={[
              'Closer to 0 is better (less drawdown).',
              'Your data may store it as negative; the UI often uses |drawdown|.',
              'Set a max drawdown threshold you can realistically tolerate.',
            ]}
          />

          <MetricItem
            name="holdout_trades"
            badge={{ label: 'reliability', color: 'gray' }}
            meaning="Number of trades that EXIT within the holdout window."
            why="More trades generally makes the results more reliable; too few trades can produce misleadingly great metrics."
            tips={[
              'Bigger sample sizes are more trustworthy.',
              'Low trade counts make Sharpe and return unstable.',
              'Use Min holdout trades to reduce “lucky winners.”',
            ]}
          />

          <MetricItem
            name="entry_min_ev"
            meaning="Minimum expected value required to enter a trade (model gate)."
            why="Controls how selective the strategy is; higher EV thresholds usually mean fewer trades but potentially higher quality."
            tips={[
              'Higher can reduce bad trades but may starve the strategy of opportunities.',
              'Look for stability: a plateau of good results across nearby EV values.',
            ]}
          />

          <MetricItem
            name="retrain_every"
            meaning="How frequently you retrain the model in walk-forward mode (in bars/days)."
            why="Too frequent retraining can overfit to short-term noise; too infrequent can become stale."
            tips={[
              'Smaller = more responsive, more compute.',
              'Larger = more stable, less compute.',
              'Prefer values that work across multiple lookbacks.',
            ]}
          />

          <MetricItem
            name="lookback"
            meaning="How many bars/days of history are used for each retrain window."
            why="Controls what regimes the model learns from. Too short can overfit; too long can dilute recent structure."
            tips={[
              'Shorter = adapts faster, risk of regime overfit.',
              'Longer = more stable, may be less responsive.',
              'Check whether winners cluster around a lookback range.',
            ]}
          />

          <MetricItem
            name="status"
            badge={{ label: 'quality', color: 'orange' }}
            meaning="Whether the config ran successfully and whether it passed filters."
            why="Helps you ignore invalid rows and avoid overfitting to tiny samples."
            tips={[
              'ok = usable result.',
              'filtered_min_trades = ran, but too few trades for reliability.',
              'error = failed execution (see Top errors).',
            ]}
          />
        </SimpleGrid>
      </Stack>
    </Modal>
  );
}
