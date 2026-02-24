import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * Catch unhandled React rendering errors and display a fallback UI
 * instead of an empty white screen.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // eslint-disable-next-line no-console
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <div className="flex min-h-screen items-center justify-center bg-slate-950 p-8 text-center text-white">
          <div>
            <h1 className="mb-2 text-2xl font-bold">Something went wrong</h1>
            <p className="mb-4 text-slate-400">
              {this.state.error?.message ?? "An unexpected error occurred."}
            </p>
            <button
              className="rounded bg-cyan-600 px-4 py-2 text-sm font-medium hover:bg-cyan-500"
              onClick={() => this.setState({ hasError: false, error: null })}
            >
              Try again
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
