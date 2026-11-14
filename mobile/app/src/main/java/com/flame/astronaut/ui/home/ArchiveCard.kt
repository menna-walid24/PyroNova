package com.flame.astronaut.ui.home

import android.content.Context
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.flame.astronaut.ui.theme.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * Home "research archive" card built from the NASA PSI dashboard data bundled in assets
 * (psi_summary.json, saffire3_timeline.json). It is archived data, never presented as live telemetry.
 */
data class PsiArchive(
    val investigations: Int,
    val publications: Int,
    val tests: Int,
    val yearFrom: Int?,
    val yearTo: Int?,
    val smoke: List<Float>,      // SAFFIRE-III smoke obscuration, 12 s buckets
    val igniterOn: List<Boolean>,
    val durationS: Int,
)

private fun loadArchive(context: Context): PsiArchive? = runCatching {
    val s = JSONObject(context.assets.open("psi_summary.json").bufferedReader().use { it.readText() })
    val years = s.optJSONArray("year_range")
    val t = JSONArray(context.assets.open("saffire3_timeline.json").bufferedReader().use { it.readText() })
    val smoke = ArrayList<Float>(t.length())
    val ign = ArrayList<Boolean>(t.length())
    var last = 0.0
    for (i in 0 until t.length()) {
        val o = t.getJSONObject(i)
        smoke += if (o.isNull("smoke_f2_obscuration")) 0f else o.getDouble("smoke_f2_obscuration").toFloat()
        ign += o.optInt("igniter_on", 0) == 1
        last = o.optDouble("t_s", last)
    }
    PsiArchive(
        investigations = s.optInt("investigations"),
        publications = s.optInt("publications"),
        tests = s.optInt("ground_and_flight_tests"),
        yearFrom = years?.optInt(0),
        yearTo = years?.optInt(1),
        smoke = smoke,
        igniterOn = ign,
        durationS = last.toInt(),
    )
}.getOrNull()

@Composable
fun ArchiveCardContent() {
    val context = LocalContext.current
    val archive by produceState<PsiArchive?>(null) { value = withContext(Dispatchers.IO) { loadArchive(context) } }
    val a = archive
    if (a == null) {
        Text("Loading NASA research archive…", style = sans(14.5f, C.Muted), modifier = Modifier.padding(vertical = 18.dp))
        return
    }
    Spacer(Modifier.height(14.dp))
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        Stat("${a.investigations}", "INVESTIGATIONS", Modifier.weight(1f))
        Stat("${a.publications}", "PUBLICATIONS", Modifier.weight(1f))
        Stat("${a.tests}", "TESTS", Modifier.weight(1f))
    }
    Spacer(Modifier.height(16.dp))
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("SAFFIRE-III · SMOKE SIGNAL", style = mono(10.5f, C.Muted, .14f), modifier = Modifier.weight(1f))
        Text("${a.durationS / 60} min · replay", style = mono(10.5f, C.Faint, .04f))
    }
    Spacer(Modifier.height(8.dp))
    Sparkline(a, Modifier.fillMaxWidth().height(64.dp))
    Spacer(Modifier.height(8.dp))
    val years = if (a.yearFrom != null && a.yearTo != null) " · ${a.yearFrom}–${a.yearTo}" else ""
    Text(
        "Archived NASA PSI data$years. Raw signal, unit not stated in the source; orange marks show the igniter on. Not live cabin telemetry.",
        style = sans(12.5f, C.Faint, lh = 1.35f),
    )
}

@Composable
private fun Stat(value: String, label: String, modifier: Modifier) {
    Column(
        modifier.clip(RoundedCornerShape(12.dp)).background(C.Surface3).padding(vertical = 10.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(value, style = chakra(22f, C.Text, FontWeight.Bold))
        Text(label, style = mono(8.5f, C.Muted, .1f), maxLines = 1)
    }
}

@Composable
private fun Sparkline(a: PsiArchive, modifier: Modifier) {
    Canvas(modifier.clip(RoundedCornerShape(10.dp)).border(1.dp, C.Stroke, RoundedCornerShape(10.dp))) {
        val n = a.smoke.size
        if (n < 2) return@Canvas
        val max = (a.smoke.maxOrNull() ?: 1f).coerceAtLeast(0.001f)
        val pad = 6.dp.toPx()
        val w = size.width
        val h = size.height - pad * 2
        fun x(i: Int) = w * i / (n - 1)
        fun y(v: Float) = pad + h * (1f - v / max)
        a.igniterOn.forEachIndexed { i, on ->
            if (on) drawLine(C.Orange.copy(alpha = .8f), Offset(x(i), 0f), Offset(x(i), size.height), 2.dp.toPx())
        }
        val p = Path().apply {
            moveTo(x(0), y(a.smoke[0]))
            for (i in 1 until n) lineTo(x(i), y(a.smoke[i]))
        }
        drawPath(p, Color(0xFF6FC3FF), style = Stroke(1.5.dp.toPx()))
    }
}
