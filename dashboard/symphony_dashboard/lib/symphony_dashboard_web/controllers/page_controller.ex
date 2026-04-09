defmodule SymphonyDashboardWeb.PageController do
  use SymphonyDashboardWeb, :controller

  def home(conn, _params) do
    render(conn, :home)
  end
end
