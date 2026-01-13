import { Card, Group, Text, Title } from '@mantine/core';

export default function MetricCard(props: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <Card withBorder radius="md" p="md">
      <Text size="sm" c="dimmed">
        {props.label}
      </Text>
      <Group justify="space-between" align="end" mt={6}>
        <Title order={3}>{props.value}</Title>
      </Group>
      {props.sub ? (
        <Text size="xs" c="dimmed" mt={6}>
          {props.sub}
        </Text>
      ) : null}
    </Card>
  );
}
