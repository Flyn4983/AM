/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        abyss: {
          950: "#06090e",
          900: "#0a0e14",
          800: "#0f141b",
          700: "#161d28",
          600: "#1e2733",
          500: "#2a3544",
        },
        laser: {
          50: "#fff3ec",
          100: "#ffe2cf",
          300: "#ff9d6b",
          400: "#ff7a43",
          500: "#ff6b35",
          600: "#eb5016",
        },
        gold: {
          300: "#ffd97a",
          400: "#ffc93c",
          500: "#f5b400",
        },
        coolant: {
          300: "#8fdcff",
          400: "#4fc3f7",
          500: "#29b6f6",
        },
        deposit: {
          300: "#9bf7e3",
          400: "#64ffda",
          500: "#33d4ae",
        },
        ink: {
          100: "#eceef2",
          200: "#d4d9e0",
          300: "#a7b0bd",
          400: "#6b7785",
          500: "#45505e",
        },
      },
      fontFamily: {
        mono: ["Space Mono", "ui-monospace", "monospace"],
        display: ["Instrument Serif", "Georgia", "serif"],
        sans: ["Inter", "ui-sans-serif", "system-ui"],
      },
      boxShadow: {
        laser: "0 0 12px rgba(255,107,53,0.55), 0 0 40px rgba(255,107,53,0.18)",
        pool: "0 0 8px rgba(255,201,60,0.6), inset 0 0 12px rgba(255,107,53,0.3)",
        glow: "0 0 0 1px rgba(100,255,218,0.15), 0 0 24px rgba(100,255,218,0.08)",
      },
      backgroundImage: {
        grid: "linear-gradient(rgba(100,255,218,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(100,255,218,0.06) 1px, transparent 1px)",
        scanline: "repeating-linear-gradient(0deg, rgba(255,255,255,0.02) 0 1px, transparent 1px 3px)",
      },
      animation: {
        pulseLaser: "pulseLaser 1.6s ease-in-out infinite",
        scanH: "scanH 3.2s linear infinite",
        flicker: "flicker 2.8s linear infinite",
      },
      keyframes: {
        pulseLaser: {
          "0%,100%": { boxShadow: "0 0 8px rgba(255,107,53,0.45), 0 0 30px rgba(255,107,53,0.2)" },
          "50%": { boxShadow: "0 0 16px rgba(255,107,53,0.85), 0 0 60px rgba(255,107,53,0.35)" },
        },
        scanH: {
          "0%": { transform: "translateX(-120%)" },
          "100%": { transform: "translateX(120%)" },
        },
        flicker: {
          "0%,100%": { opacity: 1 },
          "45%": { opacity: 0.92 },
          "50%": { opacity: 0.7 },
          "55%": { opacity: 0.96 },
        },
      },
    },
  },
  plugins: [],
};
