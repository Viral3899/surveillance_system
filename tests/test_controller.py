from surveillance.system import SurveillanceController


def test_controller_reports_stopped_status():
    controller = SurveillanceController(camera_id=0, gui_mode=False)
    status = controller.get_status()
    assert status["status"] == "stopped"
