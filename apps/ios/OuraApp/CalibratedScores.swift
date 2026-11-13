import SwiftUI
import UniformTypeIdentifiers

// Oura-calibrated Readiness / Activity (`calibrated_scores`, crates/oura-scores).
// SIBLING CLIENT: the web dashboard renders the same card (`renderCalibrated` in
// dashboard/web/app.js) — keep the two in step.

private let contributorLabel: [String: String] = [
    "activity_balance": "Activity balance", "body_temperature": "Body temperature",
    "hrv_balance": "HRV balance", "previous_day_activity": "Previous day activity",
    "previous_night": "Previous night", "recovery_index": "Recovery index",
    "resting_heart_rate": "Resting heart rate", "sleep_balance": "Sleep balance",
    "meet_daily_targets": "Meet daily targets", "move_every_hour": "Move every hour",
    "recovery_time": "Recovery time", "stay_active": "Stay active",
    "training_frequency": "Training frequency", "training_volume": "Training volume",
]

private func scoreTint(_ v: Double) -> Color {
    switch v {
    case 85...: return Obs.good
    case 70..<85: return Obs.ink
    default: return Obs.bad
    }
}

/// "2026-08-27" → "Aug 27".
private func monthDay(_ ymd: String) -> String {
    let parts = ymd.split(separator: "-")
    guard parts.count == 3, let m = Int(parts[1]), let d = Int(parts[2]), (1...12).contains(m) else { return ymd }
    let months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return "\(months[m - 1]) \(d)"
}

struct CalibratedScoresCard: View {
    let scores: CalibratedScores
    @State private var expanded = false

    private enum Kind { case readiness, activity }

    private func scored(_ kind: Kind) -> [(date: String, score: CalibratedScore)] {
        scores.days.compactMap { day in
            let s = kind == .readiness ? day.readiness : day.activity
            return s.map { (day.date, $0) }
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                ObsTag("readiness & activity", icon: "figure.run")
                Spacer()
                Text("calibrated on your oura history").font(Obs.mono(9)).foregroundStyle(Obs.muted)
            }
            if let error = scores.error {
                Text("Could not load the calibrated scores: \(error)")
                    .font(Obs.mono(11)).foregroundStyle(Obs.bad)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                HStack(alignment: .top, spacing: 20) {
                    column("Readiness", scored(.readiness), empty: "Needs a night of sleep.")
                    column("Activity", scored(.activity), empty: "Needs a day of activity.")
                }
                if expanded {
                    contributors("Readiness", scored(.readiness).last?.score)
                    contributors("Activity", scored(.activity).last?.score)
                }
                Button { expanded.toggle() } label: {
                    Text(expanded ? "hide contributors" : "show contributors")
                        .font(Obs.mono(10)).foregroundStyle(Obs.ink2)
                }
                .buttonStyle(.plain)
            }
        }
    }

    private func column(_ title: String, _ days: [(date: String, score: CalibratedScore)], empty: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(Obs.mono(11)).foregroundStyle(Obs.ink2)
            if let last = days.last {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("\(Int(last.score.score))")
                        .font(Obs.mono(32, .medium)).foregroundStyle(scoreTint(last.score.score)).monospacedDigit()
                    Text(monthDay(last.date)).font(Obs.mono(10)).foregroundStyle(Obs.muted)
                }
                // last 14 scored days, oldest first; scores live in ~40–100, so the bars
                // are scaled over that band to make day-to-day change visible
                HStack(alignment: .bottom, spacing: 2) {
                    ForEach(Array(days.suffix(14).enumerated()), id: \.offset) { _, day in
                        RoundedRectangle(cornerRadius: 1.5)
                            .fill(scoreTint(day.score.score).opacity(0.75))
                            .frame(height: max(2, min(28, (day.score.score - 40) / 60 * 28)))
                    }
                }
                .frame(height: 28, alignment: .bottom)
                .accessibilityLabel("\(title), last \(min(days.count, 14)) days")
            } else {
                Text(empty).font(Obs.mono(11)).foregroundStyle(Obs.muted)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func contributors(_ title: String, _ score: CalibratedScore?) -> some View {
        if let score {
            VStack(alignment: .leading, spacing: 7) {
                Text(title.uppercased()).font(Obs.mono(9, .medium)).tracking(1.2).foregroundStyle(Obs.muted)
                // lowest first: what is holding the score back
                ForEach(score.contributors.sorted { $0.value < $1.value }, id: \.key) { name, value in
                    VStack(alignment: .leading, spacing: 3) {
                        HStack {
                            Text(contributorLabel[name] ?? name).font(Obs.mono(11)).foregroundStyle(Obs.ink2)
                            Spacer()
                            Text("\(Int(value))").font(Obs.mono(11, .medium)).foregroundStyle(Obs.ink).monospacedDigit()
                        }
                        GeometryReader { geo in
                            ZStack(alignment: .leading) {
                                Rectangle().fill(Obs.rule).frame(height: 3)
                                Rectangle().fill(scoreTint(value))
                                    .frame(width: geo.size.width * min(100, max(0, value)) / 100, height: 3)
                            }
                        }
                        .frame(height: 3)
                    }
                }
            }
        }
    }
}

/// The personal score model (`export_score_params.json`, written on the Mac by
/// `tools/fit_export_scores.py`). The core reads it from next to the DB, so importing
/// copies it there; it never ships inside the app.
enum CalibrationParams {
    static let fileName = "export_score_params.json"

    static var url: URL { DB.url.deletingLastPathComponent().appendingPathComponent(fileName) }

    static var isInstalled: Bool { FileManager.default.fileExists(atPath: url.path) }

    /// The export's date span recorded in the file, e.g. "2022-11-21 – 2026-08-27".
    static var coverage: String? {
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let days = json["days"] as? [String], days.count == 2 else { return nil }
        return "\(days[0]) – \(days[1])"
    }

    enum ImportError: LocalizedError {
        case unreadable, notParams
        var errorDescription: String? {
            switch self {
            case .unreadable: return "Could not read that file."
            case .notParams: return "That is not an export_score_params.json from tools/fit_export_scores.py."
            }
        }
    }

    static func install(from source: URL) throws {
        let scoped = source.startAccessingSecurityScopedResource()
        defer { if scoped { source.stopAccessingSecurityScopedResource() } }
        guard let data = try? Data(contentsOf: source) else { throw ImportError.unreadable }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              json["version"] is Int, let scores = json["scores"] as? [String: Any],
              scores["readiness"] != nil, scores["activity"] != nil else { throw ImportError.notParams }
        try data.write(to: url, options: [.atomic, .completeFileProtection])
    }

    static func remove() throws {
        if isInstalled { try FileManager.default.removeItem(at: url) }
    }
}

/// Settings section: import or remove the calibration file.
struct CalibrationSettingsSection: View {
    let onChanged: () -> Void
    @State private var picking = false
    @State private var installed = CalibrationParams.isInstalled
    @State private var message: String?

    var body: some View {
        Section {
            Button {
                message = nil; picking = true
            } label: {
                Label(installed ? "Replace calibration file" : "Import calibration file", systemImage: "square.and.arrow.down")
            }
            if installed {
                Button("Remove calibration") {
                    do {
                        try CalibrationParams.remove()
                        installed = false; message = "Removed."; onChanged()
                    } catch { message = error.localizedDescription }
                }
                .foregroundStyle(Obs.bad)
            }
            if let message { Text(message).font(.footnote).foregroundStyle(Obs.ink2) }
        } header: {
            Text("Oura-calibrated scores")
        } footer: {
            Text(installed
                 ? "Readiness and Activity are estimated with a model fitted on your Oura history\(CalibrationParams.coverage.map { " (\($0))" } ?? ""). The file stays on this iPhone."
                 : "Fit the model on your Oura export with tools/fit_export_scores.py on your Mac, then import export_score_params.json here (AirDrop or Files). The file stays on this iPhone.")
        }
        .fileImporter(isPresented: $picking, allowedContentTypes: [.json]) { result in
            switch result {
            case .success(let url):
                do {
                    try CalibrationParams.install(from: url)
                    installed = true; message = "Imported."; onChanged()
                } catch { message = error.localizedDescription }
            case .failure(let error):
                message = error.localizedDescription
            }
        }
    }
}
