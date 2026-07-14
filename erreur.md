react-dom-DZQ9hRBr.js?v=6789276a:17231 Download the React DevTools for a better development experience: https://reactjs.org/link/react-devtools
react-dom-DZQ9hRBr.js?v=6789276a:13637 Uncaught NotFoundError: Failed to execute 'removeChild' on 'Node': The node to be removed is not a child of this node.
    at removeChildFromContainer (react-dom-DZQ9hRBr.js?v=6789276a:7329:19)
    at commitDeletionEffectsOnFiber (react-dom-DZQ9hRBr.js?v=6789276a:14233:58)
    at recursivelyTraverseDeletionEffects (react-dom-DZQ9hRBr.js?v=6789276a:14218:5)
    at commitDeletionEffectsOnFiber (react-dom-DZQ9hRBr.js?v=6789276a:14245:6)
    at recursivelyTraverseDeletionEffects (react-dom-DZQ9hRBr.js?v=6789276a:14218:5)
    at commitDeletionEffectsOnFiber (react-dom-DZQ9hRBr.js?v=6789276a:14279:6)
    at recursivelyTraverseDeletionEffects (react-dom-DZQ9hRBr.js?v=6789276a:14218:5)
    at commitDeletionEffectsOnFiber (react-dom-DZQ9hRBr.js?v=6789276a:14230:6)
    at recursivelyTraverseDeletionEffects (react-dom-DZQ9hRBr.js?v=6789276a:14218:5)
    at commitDeletionEffectsOnFiber (react-dom-DZQ9hRBr.js?v=6789276a:14279:6)
react-dom-DZQ9hRBr.js?v=6789276a:11719 The above error occurred in the <InferenceIntelligence> component:

    at InferenceIntelligence (http://localhost:8081/src/components/inference/InferenceIntelligence.tsx:7:41)
    at div
    at div
    at WorkspaceTabs (http://localhost:8081/src/components/layout/WorkspaceTabs.tsx:31:33)
    at div
    at div
    at Index (http://localhost:8081/src/pages/Index.tsx:760:31)
    at RenderedRoute (http://localhost:8081/node_modules/.vite/deps/react-router-dom.js?v=9d1dae2b:3329:8)
    at Routes (http://localhost:8081/node_modules/.vite/deps/react-router-dom.js?v=9d1dae2b:3772:8)
    at Router (http://localhost:8081/node_modules/.vite/deps/react-router-dom.js?v=9d1dae2b:3720:18)
    at BrowserRouter (http://localhost:8081/node_modules/.vite/deps/react-router-dom.js?v=9d1dae2b:4395:8)
    at Provider (http://localhost:8081/node_modules/.vite/deps/dist-BeNyY9nU.js?v=6789276a:33:12)
    at TooltipProvider (http://localhost:8081/node_modules/.vite/deps/@radix-ui_react-tooltip.js?v=6d0bd417:24:10)
    at HardwareProvider (http://localhost:8081/src/contexts/HardwareContext.tsx:131:36)
    at PlanProvider (http://localhost:8081/src/contexts/PlanContext.tsx:15:32)
    at ThemeProvider (http://localhost:8081/src/contexts/ThemeContext.tsx:33:33)
    at AuthProvider (http://localhost:8081/src/contexts/AuthContext.tsx:7:32)
    at QueryClientProvider (http://localhost:8081/node_modules/.vite/deps/@tanstack_react-query.js?v=3e3deb24:2369:30)
    at App

Consider adding an error boundary to your tree to customize error handling behavior.
Visit https://reactjs.org/link/error-boundaries to learn more about error boundaries.
logCapturedError @ react-dom-DZQ9hRBr.js?v=6789276a:11719
react-dom-DZQ9hRBr.js?v=6789276a:7810 Uncaught NotFoundError: Failed to execute 'removeChild' on 'Node': The node to be removed is not a child of this node.
    at removeChildFromContainer (react-dom-DZQ9hRBr.js?v=6789276a:7329:19)
    at commitDeletionEffectsOnFiber (react-dom-DZQ9hRBr.js?v=6789276a:14233:58)
    at recursivelyTraverseDeletionEffects (react-dom-DZQ9hRBr.js?v=6789276a:14218:5)
    at commitDeletionEffectsOnFiber (react-dom-DZQ9hRBr.js?v=6789276a:14245:6)
    at recursivelyTraverseDeletionEffects (react-dom-DZQ9hRBr.js?v=6789276a:14218:5)
    at commitDeletionEffectsOnFiber (react-dom-DZQ9hRBr.js?v=6789276a:14279:6)
    at recursivelyTraverseDeletionEffects (react-dom-DZQ9hRBr.js?v=6789276a:14218:5)
    at commitDeletionEffectsOnFiber (react-dom-DZQ9hRBr.js?v=6789276a:14230:6)
    at recursivelyTraverseDeletionEffects (react-dom-DZQ9hRBr.js?v=6789276a:14218:5)
    at commitDeletionEffectsOnFiber (react-dom-DZQ9hRBr.js?v=6789276a:14279:6)
