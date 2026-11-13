# Installing OuraApp on your own iPhone with a free Apple ID

No Apple Developer Program, no TestFlight: Xcode signs the app with the free
"Personal Team" every Apple ID has, and installs it over USB. The trade-off is that
**the install expires after 7 days**; re-running it from Xcode renews it and keeps
your data.

This builds the model-free app (`project-ci.yml`): ring pairing and sync, the summary,
the Oura-calibrated Readiness/Activity scores and Apple Health export. It leaves out
the on-device ML models (LibTorch, `.ptl` files), which are not needed for any of that.
`OuraApp/project-personal.yml` is that same spec with your own bundle id
(`cz.michal.ouraapp`) and team.

## One-time setup

1. **Xcode** (free): install it from the Mac App Store, open it once, accept the
   licence, and let it install the iOS platform. Then point the command-line tools at
   it (asks for your password):
   ```bash
   sudo xcode-select -s /Applications/Xcode.app
   ```
2. **Your Apple ID in Xcode**: Xcode → Settings → Accounts → **+** → Apple ID. This
   creates your *Personal Team*.
3. **Your Team ID**, so regenerating the project keeps your signing:
   ```bash
   defaults read com.apple.dt.Xcode IDEProvisioningTeamByIdentifier
   ```
   Use the `teamID` of the entry marked `isFreeProvisioningTeam = 1` (10 characters).
   If the key is missing, select your team once under the target's
   *Signing & Capabilities*; Xcode shows it there too.
4. **The iPhone**: connect it by USB and tap *Trust*. Enable
   Settings → Privacy & Security → **Developer Mode** (the switch appears after the
   phone has been connected to Xcode once) and restart when asked.

## Build and install

From the repository root:

```bash
./apps/ios/build-xcframework.sh                      # Rust core for device + simulator
cd apps/ios/OuraApp
OURA_TEAM_ID=XXXXXXXXXX xcodegen generate --spec project-personal.yml
open OuraApp.xcodeproj
```

In Xcode pick your iPhone as the run destination and press **Run** (⌘R). On the first
launch iOS blocks the app until you trust it: Settings → General →
**VPN & Device Management** → your Apple ID → *Trust*.

Re-run `build-xcframework.sh` after changing Rust code, and `xcodegen generate` after
pulling changes that add or remove Swift files.

## Every 7 days

Connect the phone, open the project, press **Run**. The app's data (the synced
database, the calibration file) survives a reinstall. **Deleting** the app does not:
export the ring key to your password manager first (Phase 2).

## Load your Oura calibration

On the Mac the fit writes `local/export_score_params.json`
(`tools/fit_export_scores.py`). AirDrop it to the iPhone (or save it to Files), then in
the app: profile → **Oura-calibrated scores** → *Import calibration file*. The home
screen then shows the Readiness & activity card once the ring has synced.

## Free-tier limits

- 7-day expiry; at most 3 sideloaded apps per device; a limited number of new bundle
  ids per week, so keep `cz.michal.ouraapp` rather than renaming it often.
- If Xcode refuses to sign because of a capability, it is HealthKit (the only one the
  app requests; Bluetooth background sync is a plain background mode and works). Then
  remove the `com.apple.developer.healthkit` key from `OuraApp.entitlements` for your
  local build; everything except Apple Health export keeps working.
- Switching to the paid program later only changes `OURA_TEAM_ID` (and unlocks
  TestFlight, see `TESTFLIGHT.md`).
