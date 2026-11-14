package com.flame.astronaut.data.real

import android.content.Context
import com.flame.astronaut.BuildConfig
import com.flame.astronaut.assistant.AssistantJson
import com.flame.astronaut.assistant.AskResponse
import com.flame.astronaut.domain.*
import com.flame.astronaut.net.*
import com.flame.astronaut.push.PushTokens
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.*
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.util.concurrent.TimeUnit

/** runCatching that never swallows coroutine cancellation. */
private inline fun <T> call(block: () -> T): Result<T> = try {
    Result.success(block())
} catch (e: CancellationException) {
    throw e
} catch (e: Exception) {
    Result.failure(e)
}

private fun UserDto.toProfile() = UserProfile(
    name = name.orEmpty(),
    email = email.orEmpty(),
    module = (assignedModule ?: module).orEmpty(),
)

// ------------------------------------------------------------------ auth

class RealAuthRepository(
    private val api: PyronovaApi,
    private val tokens: TokenStore,
    private val push: PushTokens,
    scope: CoroutineScope,
) : AuthRepository {

    override val user: StateFlow<UserProfile?> = tokens.user.stateIn(scope, SharingStarted.Eagerly, null)

    override fun isLoggedIn() = tokens.currentToken() != null

    override suspend fun register(
        name: String, email: String, password: String, passwordConfirmation: String, module: AssignedModule,
    ): Result<String> = call {
        val r = api.register(RegisterRequest(name.trim(), email.trim(), password, passwordConfirmation, module.name))
        r.verificationId ?: error(r.message ?: "Registration could not be started.")
    }

    override suspend fun verifyOtp(verificationId: String, otp: String): Result<Unit> = call {
        completeSignIn(api.verifyOtp(VerifyOtpRequest(verificationId, otp.trim())))
    }

    override suspend fun login(email: String, password: String): Result<Unit> = call {
        completeSignIn(api.login(LoginRequest(email.trim(), password)))
    }

    /** Saves token + profile, then registers this device for push. */
    private suspend fun completeSignIn(r: AuthResponse) {
        val token = r.token ?: error(r.message ?: "Sign-in failed.")
        tokens.save(token, r.user?.toProfile() ?: UserProfile("", "", ""))
        syncFcmToken()
    }

    override suspend fun logout() {
        runCatching { api.logout() }
        tokens.clear()
    }

    override suspend fun forgotPassword(email: String): Result<String> = call {
        val r = api.forgotPassword(ForgotPasswordRequest(email.trim()))
        r.status ?: r.message ?: "If that email exists, a reset link has been sent."
    }

    override suspend fun resetPassword(token: String, email: String, password: String, passwordConfirmation: String): Result<String> = call {
        val r = api.resetPassword(ResetPasswordRequest(otp = token.trim(), token = token.trim(), email = email.trim(), password = password, passwordConfirmation = passwordConfirmation))
        r.status ?: r.message ?: "Your password has been reset."
    }

    override suspend fun refreshProfile(): Result<UserProfile> = call {
        val p = api.me().toProfile()
        tokens.saveUser(p)
        p
    }

    override suspend fun resetModule(module: AssignedModule): Result<Unit> = call {
        api.resetModule(ResetModuleRequest(module.name))
        user.value?.let { tokens.saveUser(it.copy(module = module.name)) }
    }

    override suspend fun syncFcmToken() {
        if (!isLoggedIn()) return
        val t = push.fetch() ?: return
        runCatching { api.updateFcmToken(FcmTokenRequest(t)) }
    }
}

// ------------------------------------------------------------------ AI assistant (server-side history)

class RealChatRepository(private val api: PyronovaApi) : ChatRepository {

    override suspend fun chats(): Result<List<ChatSummary>> = call {
        api.chats().chats.orEmpty()
            .sortedByDescending { it.updatedAt.orEmpty() }
            .map { ChatSummary(it.id, it.title, it.updatedAt) }
    }

    override suspend fun chat(id: Long): Result<ChatDetail> = call {
        val c = api.chat(id).chat ?: error("Chat not found.")
        ChatDetail(
            id = c.id,
            title = c.title,
            turns = c.messages.orEmpty().map { ChatTurn(it.id ?: 0, it.message.orEmpty(), it.answer) },
        )
    }

    override suspend fun createChat(): Result<Long> = call {
        api.createChat().chat?.id ?: error("Couldn't start a new chat.")
    }

    override suspend fun ask(question: String, chatId: Long): Result<AskResult> = call {
        val raw = api.ask(AskBody(question, chatId)).string()
        if (com.flame.astronaut.BuildConfig.DEBUG) android.util.Log.d("FlameAsk", raw.take(3500))
        val root = AssistantJson.instance.parseToJsonElement(raw.trim().removePrefix("﻿")).jsonObject
        val answerEl = (root["data"] as? JsonObject)?.get("answer") ?: root["answer"] ?: root["data"]
        val answer = answerEl?.let(::lenientAskResponse)
            ?: error(root["message"]?.jsonPrimitive?.contentOrNull ?: "The assistant returned no answer.")
        val newChatId = root["chat_id"]?.jsonPrimitive?.contentOrNull?.toLongOrNull() ?: chatId
        AskResult(newChatId, root["title"]?.jsonPrimitive?.contentOrNull, answer)
    }

    /**
     * The AI's JSON shape changes with its data. Keep every field that still parses and drop the ones whose type
     * changed, so the answer text always shows instead of failing the whole reply.
     */
    private fun lenientAskResponse(el: JsonElement): AskResponse? {
        val json = AssistantJson.instance
        if (el is JsonPrimitive) {
            val text = el.contentOrNull?.takeIf { it.isNotBlank() } ?: return null
            return AssistantJson.parseOrNull(text) ?: AskResponse(answer = text)
        }
        val obj = el as? JsonObject ?: return null
        runCatching { return json.decodeFromJsonElement(AskResponse.serializer(), obj) }
        val usable = obj.filter { (k, v) -> runCatching { json.decodeFromJsonElement(AskResponse.serializer(), JsonObject(mapOf(k to v))) }.isSuccess }
        return runCatching { json.decodeFromJsonElement(AskResponse.serializer(), JsonObject(usable)) }.getOrNull()
            ?.takeIf { !it.answer.isNullOrBlank() || usable.isNotEmpty() }
    }
}

// ------------------------------------------------------------------ emergency guide (steps)

class RealGuideRepository(private val context: Context, private val api: PyronovaApi) : GuideRepository {
    private val cacheFile get() = File(context.filesDir, "guide_cache.json")

    override suspend fun getGuide(): GuideContent = withContext(Dispatchers.IO) {
        val steps = runCatching { api.guide().string() }.getOrNull()?.let { raw ->
            parse(raw)?.also { runCatching { cacheFile.writeText(raw) } } // cache only responses that parsed
        }
            ?: runCatching { cacheFile.takeIf { it.exists() }?.readText() }.getOrNull()?.let(::parse)
            ?: runCatching { context.assets.open("guide_fallback.json").bufferedReader().use { it.readText() } }.getOrNull()?.let(::parse)
            ?: emptyList()
        GuideContent(fireSteps = steps)
    }

    private fun parse(raw: String): List<Step>? = runCatching {
        AssistantJson.instance.decodeFromString(ListSerializer(GuideStepDto.serializer()), raw)
            .sortedBy { it.stepOrder ?: Int.MAX_VALUE }
            .mapNotNull { if (it.title.isNullOrBlank()) null else Step(it.title, it.body.orEmpty()) }
    }.getOrNull()?.takeIf { it.isNotEmpty() }
}

// ------------------------------------------------------------------ emergency alerts (FCM) + demo trigger

/**
 * Alerts arrive as FCM pushes ([onPushAlert]). POST /simulate asks the backend to send one to this module's
 * devices, so the same path as a real detection is exercised.
 */
class RealEmergencyRepository(
    private val api: PyronovaApi,
    private val scope: CoroutineScope,
    private val keepAwake: (Long) -> Unit = {},
) : EmergencyRepository {

    private val _alerts = MutableSharedFlow<AlertEvent>(extraBufferCapacity = 4, onBufferOverflow = BufferOverflow.DROP_OLDEST)
    override val alerts: Flow<AlertEvent> = _alerts.asSharedFlow()

    private val _current = MutableStateFlow<FireAlert?>(null)
    override val currentAlert: StateFlow<FireAlert?> = _current.asStateFlow()

    private val _scheduledAt = MutableStateFlow<Long?>(null)
    override val scheduledAt: StateFlow<Long?> = _scheduledAt.asStateFlow()
    private var scheduleJob: Job? = null

    private val _notices = MutableSharedFlow<String>(extraBufferCapacity = 4, onBufferOverflow = BufferOverflow.DROP_OLDEST)
    override val notices: Flow<String> = _notices.asSharedFlow()

    private val seen: MutableSet<String> = java.util.concurrent.ConcurrentHashMap.newKeySet()

    override fun onPushAlert(alert: FireAlert) {
        if (!seen.add(alert.id)) return
        _current.value = alert
        _alerts.tryEmit(AlertEvent(alert, AlertPresentation.FULL_SCREEN))
    }

    override suspend fun simulate(): Result<String> = call {
        val r = api.simulate()
        r.message ?: "Fire simulation triggered."
    }.recoverCatching { throw IllegalStateException(it.userMessage().friendlySimulateMessage(), it) }

    override fun scheduleAlert(delayMs: Long) {
        scheduleJob?.cancel()
        keepAwake(delayMs + 15_000)
        _scheduledAt.value = System.currentTimeMillis() + delayMs
        scheduleJob = scope.launch {
            delay(delayMs)
            _scheduledAt.value = null
            simulate()
                .onSuccess { _notices.tryEmit("Simulation sent — the alert should arrive as a push notification.") }
                .onFailure { _notices.tryEmit(it.message ?: "Couldn't trigger the simulation.") }
        }
    }

    override fun cancelScheduled() {
        scheduleJob?.cancel()
        _scheduledAt.value = null
    }

    override suspend fun resolve() {
        _current.value = null // the backend has no resolve endpoint
    }
}

// ------------------------------------------------------------------ health (splash status lines)

/** The backend has no health endpoint: "reachable" = it answered with any HTTP status below 500. */
class RealHealthRepository : HealthRepository {
    private val client = OkHttpClient.Builder().callTimeout(6, TimeUnit.SECONDS).build()

    override suspend fun check(): HealthStatus = withContext(Dispatchers.IO) {
        val ok = runCatching {
            client.newCall(Request.Builder().url(BuildConfig.BASE_URL).header("Accept", "application/json").build()).execute().use { it.code < 500 }
        }.getOrDefault(false)
        HealthStatus(backendOk = ok, aiOnline = ok)
    }
}

/**
 * Polls GET /risk/current every 5 s. The backend replays REAL archived NASA SAFFIRE-III flight telemetry
 * (not a live feed — `is_live = false`, shown on the card). If the endpoint fails, nothing is emitted, so the
 * Home screen keeps its "telemetry isn't connected" text instead of showing invented numbers.
 */
class RealRiskRepository(private val api: PyronovaApi) : RiskRepository {
    override fun observeRisk(): Flow<RiskState> = flow {
        while (true) {
            call { api.riskCurrent() }.onSuccess { d ->
                emit(
                    RiskState(
                        index = d.index.coerceIn(0, 100),
                        level = when (d.level?.uppercase()) {
                            "HIGH" -> RiskLevel.HIGH
                            "MED", "MEDIUM" -> RiskLevel.MED
                            else -> RiskLevel.LOW
                        },
                        o2 = d.o2,
                        pressure = d.pressure,
                        airflow = d.airflow ?: 0.0,
                        watchMaterial = d.watch ?: "—",
                        trend = d.trend,
                        updatedSecondsAgo = d.updatedSecondsAgo,
                        trendText = d.trendText.orEmpty(),
                        heat = d.radiantSignal,
                        igniterOn = d.igniterOn,
                        source = if (d.isLive) "" else "SAFFIRE-III flight replay · archived NASA data",
                        isLive = d.isLive,
                    ),
                )
            }
            delay(5_000)
        }
    }
}
