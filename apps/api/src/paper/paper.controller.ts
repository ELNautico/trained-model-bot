import { Controller, Get } from '@nestjs/common';
import { PaperService } from './paper.service';

@Controller('paper')
export class PaperController {
  constructor(private readonly paper: PaperService) {}

  @Get('summary')
  getSummary() {
    return this.paper.getSummary();
  }

  @Get('positions')
  getPositions() {
    return this.paper.getPositions();
  }

  @Get('trades')
  getTrades() {
    return this.paper.getTrades();
  }

  @Get('equity')
  getEquity() {
    return this.paper.getEquity();
  }
}
