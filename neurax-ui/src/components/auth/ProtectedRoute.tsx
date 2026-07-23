import { Navigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext.tsx';

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
}
