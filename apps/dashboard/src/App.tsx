import { AppShell, Group, Title, Anchor, Text } from '@mantine/core';
import { Routes, Route, Navigate, Link } from 'react-router-dom';
import RunsListPage from './pages/RunsListPage';
import RunDetailPage from './pages/RunDetailPage';
import PaperPage from './pages/PaperPage';

export default function App() {
  return (
    <AppShell header={{ height: 56 }} padding="md">
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Group gap="md">
            <Title order={4}>Trading Research Dashboard</Title>
            <Anchor component={Link} to="/runs">
              Runs
            </Anchor>
            <Anchor component={Link} to="/paper">
              Paper Trading
            </Anchor>
          </Group>

          <Text size="sm" c="dimmed">
            API: <code>/api</code>
          </Text>
        </Group>
      </AppShell.Header>

      <AppShell.Main>
        <Routes>
          <Route path="/" element={<Navigate to="/runs" replace />} />
          <Route path="/runs" element={<RunsListPage />} />
          <Route path="/runs/:runId" element={<RunDetailPage />} />
          <Route path="/paper" element={<PaperPage />} />
          <Route path="*" element={<Navigate to="/runs" replace />} />
        </Routes>
      </AppShell.Main>
    </AppShell>
  );
}
