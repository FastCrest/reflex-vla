/**
 * Reflex episode data upload endpoint — Cloudflare Worker + R2.
 *
 * Accepts anonymized episode data uploads from opt-in contributors.
 * Stores raw parquet/jsonl files in R2; tracks upload metadata in D1.
 *
 * R2 layout: reflex-raw-episodes/{contributor_hash}/{date}/{episode_id}.parquet
 *
 * Endpoints:
 *   POST   /v1/episodes/upload        — direct upload (body = gzipped file)
 *   POST   /v1/episodes/upload-url    — get presigned upload URL
 *   GET    /v1/contributor/{hash}/stats — per-contributor stats
 *   GET    /v1/stats                   — global stats
 *   GET    /healthz                    — health check
 *
 * Privacy: uploads MUST include X-Anonymized: true header.
 * Contributor identity is SHA256(machine_fingerprint)[:16], not reversible.
 *
 * Deploy:
 *   cd infra/data-worker
 *   wrangler d1 create reflex-data
 *   wrangler r2 bucket create reflex-raw-episodes
 *   # Update wrangler.toml with database_id + bucket binding
 *   wrangler d1 execute reflex-data --file=schema.sql
 *   wrangler deploy
 */

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;

    // Health check
    if (request.method === "GET" && path === "/healthz") {
      return jsonResponse({ status: "ok", service: "reflex-data-worker" });
    }

    // Global stats
    if (request.method === "GET" && path === "/v1/stats") {
      return handleGlobalStats(env);
    }

    // Per-contributor stats
    const contributorMatch = path.match(/^\/v1\/contributor\/([a-f0-9]{16})\/stats$/);
    if (request.method === "GET" && contributorMatch) {
      return handleContributorStats(env, contributorMatch[1]);
    }

    // Direct upload
    if (request.method === "POST" && path === "/v1/episodes/upload") {
      return handleUpload(request, env);
    }

    // Presigned upload URL
    if (request.method === "POST" && path === "/v1/episodes/upload-url") {
      return handlePresignedUrl(request, env);
    }

    return new Response("Not Found", { status: 404 });
  },
};


function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}


async function handleUpload(request, env) {
  // Validate anonymization header
  const anonymized = request.headers.get("X-Anonymized");
  if (anonymized !== "true") {
    return jsonResponse({ error: "anonymization_required", message: "X-Anonymized: true header required" }, 400);
  }

  const episodeId = request.headers.get("X-Episode-Id");
  const contributorHash = request.headers.get("X-Contributor-Hash");
  const fileHash = request.headers.get("X-File-Hash");

  if (!episodeId || !contributorHash || !fileHash) {
    return jsonResponse({ error: "missing_headers", message: "X-Episode-Id, X-Contributor-Hash, X-File-Hash required" }, 400);
  }

  // Validate contributor hash format (16 hex chars)
  if (!/^[a-f0-9]{16}$/.test(contributorHash)) {
    return jsonResponse({ error: "invalid_contributor_hash" }, 400);
  }

  const body = await request.arrayBuffer();
  if (!body || body.byteLength === 0) {
    return jsonResponse({ error: "empty_body" }, 400);
  }

  // R2 key: reflex-raw-episodes/{contributor_hash}/{date}/{episode_id}.parquet
  const date = new Date().toISOString().slice(0, 10);
  const r2Key = `reflex-raw-episodes/${contributorHash}/${date}/${episodeId}.parquet`;

  // Store in R2
  if (env.BUCKET) {
    try {
      await env.BUCKET.put(r2Key, body, {
        httpMetadata: {
          contentType: "application/octet-stream",
          contentEncoding: request.headers.get("Content-Encoding") || undefined,
        },
        customMetadata: {
          episodeId,
          contributorHash,
          fileHash,
          uploadedAt: new Date().toISOString(),
        },
      });
    } catch (e) {
      console.error("R2 put failed:", e.message);
      return jsonResponse({ error: "storage_failed" }, 500);
    }
  }

  // Track in D1
  if (env.DB) {
    try {
      await env.DB.prepare(
        `INSERT INTO uploads (episode_id, contributor_hash, file_hash, r2_key, file_size, uploaded_at)
         VALUES (?, ?, ?, ?, ?, ?)`
      )
        .bind(episodeId, contributorHash, fileHash, r2Key, body.byteLength, new Date().toISOString())
        .run();
    } catch (e) {
      console.error("D1 insert failed:", e.message);
      // R2 upload succeeded, D1 tracking is best-effort
    }
  }

  return jsonResponse({ status: "ok", episode_id: episodeId, r2_key: r2Key }, 201);
}


async function handlePresignedUrl(request, env) {
  let payload;
  try {
    payload = await request.json();
  } catch (e) {
    return jsonResponse({ error: "invalid_json" }, 400);
  }

  const { episode_id, contributor_hash, file_hash } = payload;
  if (!episode_id || !contributor_hash || !file_hash) {
    return jsonResponse({ error: "missing_fields", message: "episode_id, contributor_hash, file_hash required" }, 400);
  }

  if (!/^[a-f0-9]{16}$/.test(contributor_hash)) {
    return jsonResponse({ error: "invalid_contributor_hash" }, 400);
  }

  const date = new Date().toISOString().slice(0, 10);
  const r2Key = `reflex-raw-episodes/${contributor_hash}/${date}/${episode_id}.parquet`;

  // In a real deployment, this would generate a presigned URL via R2 API.
  // For now, return the direct upload endpoint as a fallback.
  return jsonResponse({
    upload_url: `https://data.fastcrest.workers.dev/v1/episodes/upload`,
    r2_key: r2Key,
    headers: {
      "X-Episode-Id": episode_id,
      "X-Contributor-Hash": contributor_hash,
      "X-File-Hash": file_hash,
      "X-Anonymized": "true",
    },
  });
}


async function handleContributorStats(env, contributorHash) {
  if (!env.DB) {
    return jsonResponse({ error: "database_unavailable" }, 503);
  }

  try {
    const result = await env.DB.prepare(
      `SELECT
        COUNT(*) as total_uploads,
        SUM(file_size) as total_bytes,
        MIN(uploaded_at) as first_upload,
        MAX(uploaded_at) as last_upload
       FROM uploads WHERE contributor_hash = ?`
    )
      .bind(contributorHash)
      .first();

    return jsonResponse({
      contributor_hash: contributorHash,
      total_uploads: result.total_uploads || 0,
      total_bytes: result.total_bytes || 0,
      first_upload: result.first_upload,
      last_upload: result.last_upload,
    });
  } catch (e) {
    console.error("D1 query failed:", e.message);
    return jsonResponse({ error: "query_failed" }, 500);
  }
}


async function handleGlobalStats(env) {
  if (!env.DB) {
    return jsonResponse({ error: "database_unavailable" }, 503);
  }

  try {
    const result = await env.DB.prepare(
      `SELECT
        COUNT(*) as total_uploads,
        COUNT(DISTINCT contributor_hash) as unique_contributors,
        SUM(file_size) as total_bytes,
        MIN(uploaded_at) as first_upload,
        MAX(uploaded_at) as last_upload
       FROM uploads`
    )
      .first();

    return jsonResponse({
      total_uploads: result.total_uploads || 0,
      unique_contributors: result.unique_contributors || 0,
      total_bytes: result.total_bytes || 0,
      first_upload: result.first_upload,
      last_upload: result.last_upload,
    });
  } catch (e) {
    console.error("D1 query failed:", e.message);
    return jsonResponse({ error: "query_failed" }, 500);
  }
}
