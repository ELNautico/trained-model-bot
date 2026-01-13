import { Controller, Get, Param, Res } from '@nestjs/common';
import type { Response } from 'express';
import * as fs from 'node:fs';
import { RunsService } from './runs.service';

@Controller('runs')
export class RunsController {
  constructor(private readonly runs: RunsService) {}

  @Get()
  async list() {
    return this.runs.listRuns();
  }

  @Get(':runId')
  async get(@Param('runId') runId: string) {
    return this.runs.getRun(runId);
  }

  @Get(':runId/download')
  async download(@Param('runId') runId: string, @Res() res: Response) {
    const { absPath, fileName } = await this.runs.streamRunCsv(runId);

    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', `attachment; filename="${fileName}"`);

    const stream = fs.createReadStream(absPath);
    stream.pipe(res);
  }

  // NEW: list top-k artifacts saved by sweep_backtests.py
  @Get(':runId/topk')
  async topk(@Param('runId') runId: string) {
    return this.runs.listTopK(runId);
  }

  // NEW: download a top-k artifact (trades/equity)
  @Get(':runId/topk/:rank/download/:which')
  async downloadTopK(
    @Param('runId') runId: string,
    @Param('rank') rank: string,
    @Param('which') which: string,
    @Res() res: Response,
  ) {
    const { absPath, fileName } = await this.runs.streamTopKCsv(runId, rank, which);

    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', `attachment; filename="${fileName}"`);

    const stream = fs.createReadStream(absPath);
    stream.pipe(res);
  }
}
