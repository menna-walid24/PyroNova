<?php

namespace App\Http\Controllers;

use Illuminate\Http\JsonResponse;
use Illuminate\Support\Facades\Http;

/**
 * Fire-risk status for the mobile Home card.
 *
 * Proxies the AI service (same AI_SERVICES_URL used by /ask). The AI service
 * replays REAL archived NASA SAFFIRE-III (PSI-100) flight telemetry — it is
 * not a live sensor feed, and the response says so (is_live = false).
 */
class RiskController extends Controller
{
    public function current(): JsonResponse
    {
        try {
            $response = Http::timeout(10)
                ->acceptJson()
                ->get(rtrim(config('services.ai-services.url'), '/') . '/risk/current');
            $response->throw();

            return response()->json($response->json());
        } catch (\Throwable $e) {
            return response()->json([
                'message' => 'Fire-risk telemetry is unavailable right now.',
            ], 503);
        }
    }
}
