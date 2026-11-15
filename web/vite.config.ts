import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev: the FastAPI backend runs on :8000. The UI calls "/api/*" and Vite strips
// the prefix, so "/api/calls/demo/state" -> "http://127.0.0.1:8000/calls/demo/state".
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Fixed port: api/server.py allows CORS only from localhost:5173 / 127.0.0.1:5173, so failing
    // fast beats Vite silently moving to 5174 when the UI calls the backend directly (VITE_API_BASE).
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
