use crate::payload::ViewerPayload;

const PAGE: &str = include_str!("page.html");

pub fn render_page(payload: &ViewerPayload) -> String {
    let data = serde_json::to_string(payload).expect("payload serializes");
    let escaped = data.replace('<', "\\u003c");
    PAGE.replace("__DATA__", &escaped)
}
