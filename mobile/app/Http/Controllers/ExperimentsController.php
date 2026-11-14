<?php

namespace App\Http\Controllers;

use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

/**
 * Experiment explorer page — proxies the AI service (AI_SERVICES_URL), which serves
 * the real NASA PSI investigations, the transparent relevance ranking and the
 * AI summaries. Read-only, so these routes can stay public.
 */
class ExperimentsController extends Controller
{
    private function ai(string $path, array $query = []): JsonResponse
    {
        try {
            $r = Http::timeout(20)->acceptJson()
                ->get(rtrim(config('services.ai-services.url'), '/') . $path, $query);
            return response()->json($r->json(), $r->status());
        } catch (\Throwable $e) {
            return response()->json(['message' => 'Experiment data is unavailable right now.'], 503);
        }
    }

    // GET /experiments?environment=cabin|lunar|mars&goal=all|flammability|suppression|detection|fundamentals&q=
    public function index(Request $request): JsonResponse
    {
        return $this->ai('/experiments', $request->only(['environment', 'goal', 'q']));
    }

    // GET /experiments/compare?ids=PSI-26,PSI-100&environment=cabin
    public function compare(Request $request): JsonResponse
    {
        return $this->ai('/experiments/compare', $request->only(['ids', 'environment']));
    }

    // GET /experiments/earth-impact
    public function earthImpact(): JsonResponse
    {
        return $this->ai('/experiments/earth-impact');
    }

    // GET /experiments/{id}?environment=cabin
    public function show(Request $request, string $id): JsonResponse
    {
        return $this->ai('/experiments/' . urlencode($id), $request->only(['environment']));
    }
}
