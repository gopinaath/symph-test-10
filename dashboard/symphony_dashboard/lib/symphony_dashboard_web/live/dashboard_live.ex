defmodule SymphonyDashboardWeb.DashboardLive do
  use SymphonyDashboardWeb, :live_view

  @refresh_interval 1_000
  @api_base "http://127.0.0.1:8000"

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      :timer.send_interval(@refresh_interval, self(), :refresh)
    end

    {:ok, assign(socket, snapshot: nil, error: nil, loading: true)}
  end

  @impl true
  def handle_info(:refresh, socket) do
    case fetch_snapshot() do
      {:ok, snapshot} ->
        {:noreply, assign(socket, snapshot: snapshot, error: nil, loading: false)}

      {:error, reason} ->
        {:noreply, assign(socket, error: reason, loading: false)}
    end
  end

  @impl true
  def handle_event("refresh", _params, socket) do
    case trigger_refresh() do
      :ok -> {:noreply, put_flash(socket, :info, "Refresh triggered")}
      {:error, _} -> {:noreply, put_flash(socket, :error, "Failed to trigger refresh")}
    end
  end

  defp fetch_snapshot do
    case Req.get("#{@api_base}/api/v1/state") do
      {:ok, %{status: 200, body: body}} -> {:ok, body}
      {:ok, %{status: status}} -> {:error, "API returned #{status}"}
      {:error, err} -> {:error, "Connection failed: #{inspect(err)}"}
    end
  end

  defp trigger_refresh do
    case Req.post("#{@api_base}/api/v1/refresh") do
      {:ok, %{status: 200}} -> :ok
      _ -> {:error, :refresh_failed}
    end
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div class="min-h-screen bg-base-200 p-6">
      <div class="max-w-7xl mx-auto">
        <!-- Header -->
        <div class="flex items-center justify-between mb-6">
          <div>
            <h1 class="text-3xl font-bold">Symphony Dashboard</h1>
            <p class="text-sm opacity-70">Autonomous coding orchestrator</p>
          </div>
          <button phx-click="refresh" class="btn btn-primary btn-sm">
            Force Refresh
          </button>
        </div>

        <%= if @loading do %>
          <div class="flex justify-center py-20">
            <span class="loading loading-spinner loading-lg"></span>
          </div>
        <% end %>

        <%= if @error do %>
          <div class="alert alert-warning mb-4">
            <span>⚠ {@error}</span>
          </div>
        <% end %>

        <%= if @snapshot do %>
          <!-- Stats Row -->
          <div class="stats shadow mb-6 w-full">
            <div class="stat">
              <div class="stat-title">Running Agents</div>
              <div class="stat-value text-primary">{map_size(@snapshot["running"])}</div>
            </div>
            <div class="stat">
              <div class="stat-title">Retry Queue</div>
              <div class="stat-value text-warning">{map_size(@snapshot["retry_queue"])}</div>
            </div>
            <div class="stat">
              <div class="stat-title">Completed</div>
              <div class="stat-value text-success">{@snapshot["completed_count"]}</div>
            </div>
            <div class="stat">
              <div class="stat-title">Total Tokens</div>
              <div class="stat-value text-info">{format_tokens(@snapshot["codex_totals"]["total_tokens"])}</div>
            </div>
            <div class="stat">
              <div class="stat-title">Poll Status</div>
              <div class="stat-value text-sm">
                <%= if @snapshot["poll_checking"] do %>
                  <span class="badge badge-info">Checking...</span>
                <% else %>
                  <span class="badge badge-ghost">{format_countdown(@snapshot["poll_countdown_ms"])}s</span>
                <% end %>
              </div>
            </div>
          </div>

          <!-- Running Agents Table -->
          <div class="card bg-base-100 shadow mb-6">
            <div class="card-body">
              <h2 class="card-title">Running Agents</h2>
              <%= if map_size(@snapshot["running"]) > 0 do %>
                <div class="overflow-x-auto">
                  <table class="table table-zebra">
                    <thead>
                      <tr>
                        <th>Issue</th>
                        <th>State</th>
                        <th>Session</th>
                        <th>Turns</th>
                        <th>Tokens</th>
                        <th>Last Event</th>
                        <th>Started</th>
                        <th>Host</th>
                      </tr>
                    </thead>
                    <tbody>
                      <%= for {_id, entry} <- @snapshot["running"] do %>
                        <tr>
                          <td>
                            <span class="font-mono font-bold">{entry["identifier"]}</span>
                            <br />
                            <span class="text-xs opacity-70">{truncate(entry["title"], 40)}</span>
                          </td>
                          <td><span class="badge badge-outline">{entry["state"]}</span></td>
                          <td class="font-mono text-xs">{truncate(entry["session_id"] || "-", 12)}</td>
                          <td>{entry["turn_count"]}</td>
                          <td>{format_tokens(entry["total_tokens"])}</td>
                          <td class="text-xs">{entry["last_event"] || "-"}</td>
                          <td class="text-xs">{format_time(entry["started_at"])}</td>
                          <td class="text-xs">{entry["worker_host"] || "local"}</td>
                        </tr>
                      <% end %>
                    </tbody>
                  </table>
                </div>
              <% else %>
                <p class="text-center opacity-50 py-4">No agents running</p>
              <% end %>
            </div>
          </div>

          <!-- Retry Queue -->
          <%= if map_size(@snapshot["retry_queue"]) > 0 do %>
            <div class="card bg-base-100 shadow mb-6">
              <div class="card-body">
                <h2 class="card-title">Retry Queue</h2>
                <div class="overflow-x-auto">
                  <table class="table table-zebra">
                    <thead>
                      <tr>
                        <th>Issue</th>
                        <th>Attempt</th>
                        <th>Due In</th>
                        <th>Error</th>
                        <th>Host</th>
                      </tr>
                    </thead>
                    <tbody>
                      <%= for {_id, entry} <- @snapshot["retry_queue"] do %>
                        <tr>
                          <td>
                            <span class="font-mono font-bold">{entry["identifier"]}</span>
                            <br />
                            <span class="text-xs opacity-70">{truncate(entry["title"], 40)}</span>
                          </td>
                          <td><span class="badge badge-warning">{entry["attempt"]}</span></td>
                          <td>{format_time(entry["due_at"])}</td>
                          <td class="text-xs text-error max-w-xs truncate">{entry["error"] || "-"}</td>
                          <td class="text-xs">{entry["preferred_host"] || "-"}</td>
                        </tr>
                      <% end %>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          <% end %>

          <!-- Rate Limits -->
          <%= if @snapshot["codex_rate_limits"] do %>
            <div class="card bg-base-100 shadow mb-6">
              <div class="card-body">
                <h2 class="card-title">Rate Limits</h2>
                <pre class="text-xs"><%= Jason.encode!(@snapshot["codex_rate_limits"], pretty: true) %></pre>
              </div>
            </div>
          <% end %>
        <% end %>
      </div>
    </div>
    """
  end

  defp format_tokens(nil), do: "0"
  defp format_tokens(n) when n >= 1_000_000, do: "#{Float.round(n / 1_000_000, 1)}M"
  defp format_tokens(n) when n >= 1_000, do: "#{Float.round(n / 1_000, 1)}K"
  defp format_tokens(n), do: "#{n}"

  defp format_countdown(nil), do: "?"
  defp format_countdown(ms), do: Float.round(ms / 1000, 0) |> trunc()

  defp format_time(nil), do: "-"
  defp format_time(iso) when is_binary(iso) do
    case DateTime.from_iso8601(iso) do
      {:ok, dt, _} -> Calendar.strftime(dt, "%H:%M:%S")
      _ -> iso
    end
  end

  defp truncate(nil, _), do: "-"
  defp truncate(s, max) when byte_size(s) > max, do: String.slice(s, 0, max) <> "..."
  defp truncate(s, _), do: s
end
