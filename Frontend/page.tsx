import Link from "next/link"
import { ArrowRight } from "lucide-react"

import { Button } from "@/components/ui/button"

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col bg-pink-50">
      <header className="border-b border-pink-200 bg-white px-4 py-3 shadow-sm">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <h1 className="text-xl font-bold text-pink-700">MamaPulse</h1>
          <div className="flex items-center gap-4">
            <Link href="/login">
              <Button variant="ghost" className="text-pink-700 hover:bg-pink-100 hover:text-pink-800">
                Login
              </Button>
            </Link>
            <Link href="/signup">
              <Button className="bg-pink-600 text-white hover:bg-pink-700">Sign Up</Button>
            </Link>
          </div>
        </div>
      </header>
      <main className="flex flex-1 flex-col items-center justify-center px-4 py-12">
        <div className="mx-auto max-w-4xl text-center">
          <h1 className="mb-6 text-4xl font-bold text-pink-800 sm:text-5xl md:text-6xl">
            Complete Maternal Healthcare Solution
          </h1>
          <p className="mb-8 text-lg text-pink-700 sm:text-xl">
            From preconception planning to postpartum care, we provide comprehensive support for your maternal health
            journey.
          </p>
          <div className="flex flex-col gap-4 sm:flex-row sm:justify-center">
            <Link href="/login?role=patient">
              <Button size="lg" className="w-full bg-pink-600 text-white hover:bg-pink-700 sm:w-auto">
                Patient Login
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
            <Link href="/login?role=doctor">
              <Button
                size="lg"
                variant="outline"
                className="w-full border-pink-600 text-pink-700 hover:bg-pink-100 sm:w-auto"
              >
                Doctor Login
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </div>
        </div>
      </main>
      <footer className="border-t border-pink-200 bg-white px-4 py-6">
        <div className="mx-auto max-w-7xl text-center text-sm text-pink-600">
          &copy; {new Date().getFullYear()} MaternalCare. All rights reserved.
        </div>
      </footer>
    </div>
  )
}

