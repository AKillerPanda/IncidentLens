import { useState, useMemo } from 'react';
import { Link } from 'react-router';
import { useSimulation } from '../hooks/useApi';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Progress } from './ui/progress';
import { Separator } from './ui/separator';
import {
  ArrowLeft,
  Play,
  Square,
  RotateCcw,
  Activity,
  AlertTriangle,
  Shield,
  Zap,
  Clock,
  Database,
  TrendingUp,
  Loader2,
} from 'lucide-react';

export function Simulation() {
  const { windows, status, running, error, done, start, stop, reset } = useSimulation();

  // Config state
  const [rate, setRate] = useState(500);
  const [windowSize, setWindowSize] = useState(5);
  const [maxRows, setMaxRows] = useState<string>('');

  // Derived stats
  const stats = useMemo(() => {
    if (windows.length === 0) {
      return { totalFlows: 0, totalIndexed: 0, avgAnomaly: 0, maxAnomaly: 0, totalWindows: 0 };
    }
    const totalFlows = windows.reduce((s, w) => s + w.num_flows, 0);
    const totalIndexed = windows.reduce((s, w) => s + w.num_indexed, 0);
    const scores = windows.map((w) => w.anomaly_score);
    const avgAnomaly = scores.reduce((a, b) => a + b, 0) / scores.length;
    const maxAnomaly = Math.max(...scores);
    return { totalFlows, totalIndexed, avgAnomaly, maxAnomaly, totalWindows: windows.length };
  }, [windows]);

  const latestWindow = windows.length > 0 ? windows[windows.length - 1] : null;

  const handleStart = () => {
    const rows = maxRows.trim() ? parseInt(maxRows, 10) : undefined;
    start({
      rate,
      window_size: windowSize,
      max_rows: rows && !isNaN(rows) ? rows : null,
      data_file: 'data/packets_0000.json',
    });
  };

  const getAnomalyColor = (score: number) => {
    if (score >= 0.7) return 'text-red-400';
    if (score >= 0.4) return 'text-orange-400';
    if (score >= 0.2) return 'text-yellow-400';
    return 'text-green-400';
  };

  const getAnomalyBadge = (score: number) => {
    if (score >= 0.7) return { label: 'CRITICAL', className: 'bg-red-500/10 text-red-500 border-red-500/20' };
    if (score >= 0.4) return { label: 'HIGH', className: 'bg-orange-500/10 text-orange-500 border-orange-500/20' };
    if (score >= 0.2) return { label: 'MEDIUM', className: 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20' };
    return { label: 'LOW', className: 'bg-green-500/10 text-green-500 border-green-500/20' };
  };

  return (
    <div className="min-h-screen bg-slate-950">
      {/* Header */}
      <div className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Link to="/">
                <Button variant="ghost" size="sm" className="text-slate-400 hover:text-slate-200">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Dashboard
                </Button>
              </Link>
              <Separator orientation="vertical" className="h-6 bg-slate-700" />
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-gradient-to-br from-emerald-500 to-cyan-600 rounded-lg flex items-center justify-center">
                  <Zap className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h1 className="text-slate-100 text-lg font-semibold">Real-Time Simulation</h1>
                  <p className="text-xs text-slate-400">Replay packet data through the GNN pipeline</p>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {running && (
                <Badge variant="outline" className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 animate-pulse">
                  <Activity className="w-3 h-3 mr-1" />
                  LIVE
                </Badge>
              )}
              {done && !running && (
                <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/20">
                  COMPLETE
                </Badge>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        {/* Config + Controls */}
        <Card className="mb-6 bg-slate-900 border-slate-800">
          <CardHeader>
            <CardTitle className="text-slate-100">Simulation Configuration</CardTitle>
            <CardDescription className="text-slate-400">
              Configure and run a packet replay simulation through the IncidentLens GNN pipeline
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="space-y-2">
                <Label htmlFor="rate" className="text-slate-300">Packet Rate (pps)</Label>
                <Input
                  id="rate"
                  type="number"
                  min={1}
                  max={10000}
                  value={rate}
                  onChange={(e) => setRate(Number(e.target.value))}
                  disabled={running}
                  className="bg-slate-800 border-slate-700 text-slate-100"
                />
                <p className="text-xs text-slate-500">Packets per second replay speed</p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="window" className="text-slate-300">Window Size (seconds)</Label>
                <Input
                  id="window"
                  type="number"
                  min={1}
                  max={60}
                  value={windowSize}
                  onChange={(e) => setWindowSize(Number(e.target.value))}
                  disabled={running}
                  className="bg-slate-800 border-slate-700 text-slate-100"
                />
                <p className="text-xs text-slate-500">Time window for flow aggregation</p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="maxrows" className="text-slate-300">Max Packets (optional)</Label>
                <Input
                  id="maxrows"
                  type="number"
                  min={1}
                  placeholder="All"
                  value={maxRows}
                  onChange={(e) => setMaxRows(e.target.value)}
                  disabled={running}
                  className="bg-slate-800 border-slate-700 text-slate-100 placeholder:text-slate-600"
                />
                <p className="text-xs text-slate-500">Limit packets loaded (blank = all)</p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {!running ? (
                <Button
                  onClick={handleStart}
                  className="bg-emerald-600 hover:bg-emerald-700 text-white"
                >
                  <Play className="w-4 h-4 mr-2" />
                  Start Simulation
                </Button>
              ) : (
                <Button
                  onClick={stop}
                  variant="destructive"
                >
                  <Square className="w-4 h-4 mr-2" />
                  Stop
                </Button>
              )}
              <Button
                onClick={reset}
                variant="outline"
                disabled={running}
                className="border-slate-700 text-slate-300 hover:bg-slate-800"
              >
                <RotateCcw className="w-4 h-4 mr-2" />
                Reset
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Status message */}
        {status && (
          <Card className="mb-6 bg-slate-900 border-slate-800">
            <CardContent className="py-4">
              <div className="flex items-center gap-3 text-slate-300">
                {running && <Loader2 className="w-4 h-4 animate-spin text-emerald-400" />}
                <span className="text-sm">{status}</span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Error */}
        {error && (
          <Card className="mb-6 bg-red-500/10 border-red-500/30">
            <CardContent className="py-4">
              <div className="flex items-center gap-3 text-red-400">
                <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                <span className="text-sm">{error}</span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Live Stats */}
        {windows.length > 0 && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
              <Card className="bg-slate-900 border-slate-800">
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm text-slate-400">Windows</CardTitle>
                  <Clock className="w-4 h-4 text-blue-400" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl text-slate-100">{stats.totalWindows}</div>
                  <p className="text-xs text-slate-500 mt-1">Processed</p>
                </CardContent>
              </Card>

              <Card className="bg-slate-900 border-slate-800">
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm text-slate-400">Total Flows</CardTitle>
                  <Activity className="w-4 h-4 text-purple-400" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl text-slate-100">{stats.totalFlows.toLocaleString()}</div>
                  <p className="text-xs text-slate-500 mt-1">Aggregated</p>
                </CardContent>
              </Card>

              <Card className="bg-slate-900 border-slate-800">
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm text-slate-400">Indexed</CardTitle>
                  <Database className="w-4 h-4 text-cyan-400" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl text-slate-100">{stats.totalIndexed.toLocaleString()}</div>
                  <p className="text-xs text-slate-500 mt-1">To Elasticsearch</p>
                </CardContent>
              </Card>

              <Card className="bg-slate-900 border-slate-800">
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm text-slate-400">Avg Anomaly</CardTitle>
                  <TrendingUp className="w-4 h-4 text-orange-400" />
                </CardHeader>
                <CardContent>
                  <div className={`text-2xl ${getAnomalyColor(stats.avgAnomaly)}`}>
                    {(stats.avgAnomaly * 100).toFixed(1)}%
                  </div>
                  <p className="text-xs text-slate-500 mt-1">Mean score</p>
                </CardContent>
              </Card>

              <Card className="bg-slate-900 border-slate-800">
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm text-slate-400">Peak Anomaly</CardTitle>
                  <AlertTriangle className="w-4 h-4 text-red-400" />
                </CardHeader>
                <CardContent>
                  <div className={`text-2xl ${getAnomalyColor(stats.maxAnomaly)}`}>
                    {(stats.maxAnomaly * 100).toFixed(1)}%
                  </div>
                  <p className="text-xs text-slate-500 mt-1">Highest window</p>
                </CardContent>
              </Card>
            </div>

            {/* Anomaly Score Timeline */}
            <Card className="mb-6 bg-slate-900 border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100">Anomaly Score Timeline</CardTitle>
                <CardDescription className="text-slate-400">
                  Per-window GNN anomaly scores — higher values indicate potential threats
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {windows.map((w) => {
                    const badge = getAnomalyBadge(w.anomaly_score);
                    return (
                      <div key={w.window_id} className="flex items-center gap-3">
                        <span className="text-xs text-slate-500 w-16 shrink-0 text-right">
                          W{w.window_id}
                        </span>
                        <div className="flex-1 relative">
                          <Progress
                            value={Math.min(w.anomaly_score * 100, 100)}
                            className="h-6 bg-slate-800"
                          />
                          <span className="absolute inset-0 flex items-center px-3 text-xs font-mono text-slate-200">
                            {(w.anomaly_score * 100).toFixed(1)}% — {w.num_flows} flows, {w.num_indexed} indexed
                          </span>
                        </div>
                        <Badge variant="outline" className={`${badge.className} shrink-0 text-xs`}>
                          {badge.label}
                        </Badge>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            {/* Window Detail Table */}
            <Card className="bg-slate-900 border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100">Window Results</CardTitle>
                <CardDescription className="text-slate-400">
                  Detailed per-window simulation output
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-800">
                        <th className="text-left py-3 px-4 text-slate-400 font-medium">Window</th>
                        <th className="text-right py-3 px-4 text-slate-400 font-medium">Flows</th>
                        <th className="text-right py-3 px-4 text-slate-400 font-medium">Indexed</th>
                        <th className="text-right py-3 px-4 text-slate-400 font-medium">Embeddings</th>
                        <th className="text-right py-3 px-4 text-slate-400 font-medium">Anomaly Score</th>
                        <th className="text-center py-3 px-4 text-slate-400 font-medium">Severity</th>
                      </tr>
                    </thead>
                    <tbody>
                      {windows.map((w) => {
                        const badge = getAnomalyBadge(w.anomaly_score);
                        return (
                          <tr key={w.window_id} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                            <td className="py-3 px-4 text-slate-200 font-mono">#{w.window_id}</td>
                            <td className="py-3 px-4 text-right text-slate-300">{w.num_flows}</td>
                            <td className="py-3 px-4 text-right text-slate-300">{w.num_indexed}</td>
                            <td className="py-3 px-4 text-right text-slate-300">{w.num_embeddings}</td>
                            <td className={`py-3 px-4 text-right font-mono ${getAnomalyColor(w.anomaly_score)}`}>
                              {(w.anomaly_score * 100).toFixed(2)}%
                            </td>
                            <td className="py-3 px-4 text-center">
                              <Badge variant="outline" className={`${badge.className} text-xs`}>
                                {badge.label}
                              </Badge>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </>
        )}

        {/* Empty state */}
        {!running && !done && windows.length === 0 && (
          <Card className="bg-slate-900 border-slate-800">
            <CardContent className="py-16">
              <div className="text-center text-slate-400">
                <Shield className="w-16 h-16 mx-auto mb-4 opacity-30" />
                <h3 className="text-lg font-medium text-slate-300 mb-2">Ready to Simulate</h3>
                <p className="text-sm max-w-md mx-auto">
                  Configure the simulation parameters above and click <strong>Start Simulation</strong> to
                  replay packet data through the GNN anomaly detection pipeline in real time.
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
