import { Toaster } from "@/components/ui/toaster.tsx";
import { Toaster as Sonner } from "@/components/ui/sonner.tsx";
import { TooltipProvider } from "@/components/ui/tooltip.tsx";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "@/contexts/AuthContext.tsx";
import { ApiKeyProvider } from "@/contexts/ApiKeyContext.tsx";
import { ThemeProvider } from "@/contexts/ThemeContext.tsx";
import { HardwareProvider } from "@/contexts/HardwareContext.tsx";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute.tsx";
import { isDesktop } from "@/services/desktopRuntime.ts";
import { DesktopChrome } from "@/components/desktop/TitleBar.tsx";
import { Suspense, lazy } from "react";

/**
 * Routes are loaded on demand.
 *
 * Every page used to be in the first chunk, so opening the studio also parsed
 * the landing page and the account page before anything could be drawn. They
 * are three separate screens and a session only ever starts on one of them.
 */
const Landing = lazy(() => import("./pages/Landing.tsx"));
const Index = lazy(() => import("./pages/Index.tsx"));
const Account = lazy(() => import("./pages/Account.tsx"));
const SharedReport = lazy(() => import("./pages/SharedReport.tsx"));
const NotFound = lazy(() => import("./pages/NotFound.tsx"));

/** Shown while a route's chunk loads — from local disk in the desktop app,
 *  which is fast enough that this is usually a single frame. */
function RouteFallback() {
  return <div className="min-h-screen bg-background" aria-busy="true" />;
}

const queryClient = new QueryClient();

/**
 * What `/` shows.
 *
 * On the web it is the landing page — the page that explains what NEURAX is to
 * someone who arrived from a link. Someone who has already installed the
 * desktop application does not need to be sold it, so once they have a profile
 * it opens straight into the studio. Before that it still shows the landing
 * page, because that is where the profile is created.
 */
function HomeRoute() {
  const { isAuthenticated } = useAuth();
  if (isDesktop() && isAuthenticated) {
    return <Navigate to="/app" replace />;
  }
  return <Landing />;
}

const App = () => (
  <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <ApiKeyProvider>
          <ThemeProvider>
              <HardwareProvider>
            <TooltipProvider>
              <Toaster />
              <Sonner />
              {/* Window chrome, where the platform draws none. Sits outside
                  the router because it frames the whole application rather
                  than belonging to any route. */}
              <DesktopChrome />
              <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
                <Suspense fallback={<RouteFallback />}>
                <Routes>
                  <Route path="/" element={<HomeRoute />} />
                  <Route path="/app" element={<ProtectedRoute><Index /></ProtectedRoute>} />
                  <Route path="/account" element={<ProtectedRoute><Account /></ProtectedRoute>} />
                  {/* Public by design — no ProtectedRoute. A share link must open
                      for anyone, with no account, same as the backend endpoint
                      it reads from. */}
                  <Route path="/s/:id" element={<SharedReport />} />
                  {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
                  <Route path="*" element={<NotFound />} />
                </Routes>
                </Suspense>
              </BrowserRouter>
            </TooltipProvider>
            </HardwareProvider>
        </ThemeProvider>
      </ApiKeyProvider>
    </AuthProvider>
  </QueryClientProvider>
);

export default App;
