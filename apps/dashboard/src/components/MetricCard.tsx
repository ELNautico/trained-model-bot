import { Card, Text, Title } from '@mantine/core';

export default function MetricCard(props: {
  label: string;
  value: string;
  sub?: string;
  valueColor?: string;
}) {
  return (
    <Card withBorder radius="md" p="md">
      <Text size="sm" c="dimmed">
        {props.label}
      </Text>
      <Title
        order={3}
        mt={6}
        style={props.valueColor ? { color: props.valueColor } : undefined}
      >
        {props.value}
      </Title>
      {props.sub ? (
        <Text size="xs" c="dimmed" mt={4}>
          {props.sub}
        </Text>
      ) : null}
    </Card>
  );
}
