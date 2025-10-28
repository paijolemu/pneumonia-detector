import './globals.css' // Biarkan baris ini

export const metadata = {
  title: 'Deteksi Pneumonia', // Ganti judulnya sekalian
  description: 'Website untuk deteksi pneumonia',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}