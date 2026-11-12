# Sonar Digital Twin

A browser-based educational active-sonar simulator built with React, TypeScript, Vite, and Plotly.js.

## Run locally

```powershell
npm install
npm run dev
```

Create a production build with `npm run build`.

## Simulation

The receive chain synthesizes a complex baseband chirp echo from point scatterers on a fixed five-point star, applies two-way spherical spreading and Thorp-style seawater absorption, adds deterministic complex Gaussian noise, and computes delay-and-sum beams, a pulse-replica matched filter, CA-CFAR normalization, and threshold detections.

Carrier frequency, aperture, pulse duration, pulse repetition rate, source level, receive scan fan, array weighting, CFAR settings, and detection threshold are interactive. The receive scan fan ranges from 30° to 360° in 30° steps and defaults to 30°; it controls both the beamforming angle coverage and the fan shown in the scene. Beam steering angles are spaced no farther apart than half the estimated horizontal beamwidth, so the number of receive beams varies with frequency and aperture. The angle–range grid is limited to 250,000 cells for interactive performance; if this limit requires wider beam spacing, the app displays the actual spacing and warns that it exceeds the target. Bandwidth is separately adjustable and its slider maximum is always one half of the carrier frequency. Bandwidth also sets the chirp, noise bandwidth, and reported range resolution.

The physical array is limited to 192 elements. Aperture width and element count are linked by default and can be unlinked. When linked, changing either control updates the other; changing frequency or bandwidth recalculates the array for spacing no greater than half a wavelength at the upper band edge. The integer element count is rounded up when needed, and aperture is reduced if the 192-element limit would otherwise be exceeded. When unlinked, frequency changes leave both array dimensions unchanged. Array dimensions use the complete λ/2-spaced array. For responsive browser updates, signal synthesis uses up to 96 uniformly distributed representative elements. The angle–range grid is capped at 250,000 cells; if this requires wider beam spacing than the target, the app displays the actual spacing and warns. Fast-time displays use a finite range grid, so the analytic range-resolution estimate can be finer than an individual displayed range cell at high bandwidth.

The Plotly charts support zoom, pan, and PNG export from the chart modebar.
