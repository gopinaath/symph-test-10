defmodule SymphonyDashboardWeb.Router do
  use SymphonyDashboardWeb, :router

  pipeline :browser do
    plug :accepts, ["html"]
    plug :fetch_session
    plug :fetch_live_flash
    plug :put_root_layout, html: {SymphonyDashboardWeb.Layouts, :root}
    plug :protect_from_forgery
    plug :put_secure_browser_headers
  end

  pipeline :api do
    plug :accepts, ["json"]
  end

  scope "/", SymphonyDashboardWeb do
    pipe_through :browser

    live "/", DashboardLive, :index
  end

  # Other scopes may use custom stacks.
  # scope "/api", SymphonyDashboardWeb do
  #   pipe_through :api
  # end
end
