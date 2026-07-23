import { Toaster } from "@/components/ui/toaster.tsx";
import { Toaster as Sonner } from "@/components/ui/sonner.tsx";
import { TooltipProvider } from "@/components/ui/tooltip.tsx";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "@/contexts/AuthContext.tsx";
import { ApiKeyProvider } from "@/contexts/ApiKeyContext.tsx";
import { PlanProvider } from "@/contexts/PlanContext.tsx";
import { ThemeProvider } from "@/contexts/ThemeContext.tsx";
import { HardwareProvider } from "@/contexts/HardwareContext.tsx";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute.tsx";
import Landing from "./pages/Landing.tsx";
import Index from "./pages/Index.tsx";
import Account from "./pages/Account.tsx";
import NotFound from "./pages/NotFound.tsx";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <ApiKeyProvider>
          <ThemeProvider>
            <PlanProvider>
              <HardwareProvider>
            <TooltipProvider>
              <Toaster />
              <Sonner />
              <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
                <Routes>
                  <Route path="/" element={<Landing />} />
                  <Route path="/app" element={<ProtectedRoute><Index /></ProtectedRoute>} />
                  <Route path="/account" element={<ProtectedRoute><Account /></ProtectedRoute>} />
                  {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
                  <Route path="*" element={<NotFound />} />
                </Routes>
              </BrowserRouter>
            </TooltipProvider>
            </HardwareProvider>
          </PlanProvider>
        </ThemeProvider>
      </ApiKeyProvider>
    </AuthProvider>
  </QueryClientProvider>
);

export default App;
