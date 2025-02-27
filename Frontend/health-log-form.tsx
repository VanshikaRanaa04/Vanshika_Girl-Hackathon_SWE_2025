"use client"

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

export function HealthLogForm({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Log Health Metrics</DialogTitle>
          <DialogDescription>Record your daily health measurements.</DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid gap-2">
            <Label htmlFor="weight">Weight (kg)</Label>
            <Input id="weight" type="number" step="0.1" placeholder="Enter your weight" />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="grid gap-2">
              <Label htmlFor="systolic">Blood Pressure (Systolic)</Label>
              <Input id="systolic" type="number" placeholder="Systolic" />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="diastolic">Blood Pressure (Diastolic)</Label>
              <Input id="diastolic" type="number" placeholder="Diastolic" />
            </div>
          </div>
          <div className="grid gap-2">
            <Label htmlFor="notes">Notes</Label>
            <Input id="notes" placeholder="Any additional notes" />
          </div>
        </div>
        <DialogFooter>
          <Button type="submit" className="bg-pink-600 hover:bg-pink-700">
            Save Measurements
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

