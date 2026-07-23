import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
  name?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ChartErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error(`[ChartErrorBoundary${this.props.name ? `: ${this.props.name}` : ''}]`, error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center h-full min-h-[200px] text-muted-foreground p-6">
          <AlertTriangle className="w-8 h-8 text-destructive mb-2" />
          <span className="text-xs font-semibold text-destructive mb-1">
            Chart Error{this.props.name ? ` — ${this.props.name}` : ''}
          </span>
          <span className="text-[10px] text-muted-foreground/60 mb-3 text-center max-w-[300px]">
            {this.state.error?.message ?? 'An unexpected error occurred while rendering this chart.'}
          </span>
          <button
            onClick={this.handleRetry}
            className="flex items-center gap-1.5 text-[10px] font-medium px-3 py-1.5 rounded-md bg-secondary hover:bg-secondary/80 transition-colors"
          >
            <RefreshCw className="w-3 h-3" />
            Retry
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
