import { Module } from '@nestjs/common';
import { RunsModule } from './runs/runs.module';
import { PaperModule } from './paper/paper.module';

@Module({
  imports: [RunsModule, PaperModule],
})
export class AppModule {}
