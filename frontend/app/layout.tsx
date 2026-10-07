import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SupportPilot",
  description: "AI customer support for CloudNotes",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
