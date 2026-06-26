# Kiểm thử API fusion-svc bằng Postman

Hướng dẫn từng bước cho QA/tester. Không cần lập trình. API là sidecar FastAPI
trong [`fusion_svc/app.py`](fusion_svc/app.py) (xem [`API.md`](API.md) để tham
khảo cho dev). Có **2 endpoint**: `GET /health`, `POST /fuse`.

`/fuse` gộp một **chuỗi (burst) crop của cùng một biển số** thành một ảnh biển số
đã phục dựng (PNG). Bạn upload N crop cùng kích thước; nhận lại một ảnh biển số đã gộp.

Bạn có thể **import collection dựng sẵn** (nhanh nhất) hoặc tự tạo request bằng
tay. Cả hai đều được hướng dẫn.

> **Trực tiếp vs qua main API.** Hướng dẫn này test sidecar **trực tiếp** (cổng
> `8100`), trả về **ảnh** biển số đã gộp. **Main traffic API** (cổng `7862`)
> cũng dùng các engine này qua `POST /predict/plates/multiframe` và
> `POST /predict/plates/video`, nhưng trả về **văn bản OCR (JSON)** thay vì ảnh
> và tự resize burst. Test các endpoint đó với `7862`, không phải `8100`.

---

## 1. Import collection (khuyến nghị)

1. Trong Postman: **File → Import** (hoặc nút **Import** ở góc trên bên trái).
2. Chọn file **[`fusion_svc/fusion-svc.postman_collection.json`](fusion-svc.postman_collection.json)**
   từ repo này.
3. Một collection **"fusion-svc API"** xuất hiện ở thanh bên trái với đầy đủ
   request đã dựng sẵn.
4. Đặt base URL một lần cho cả collection:
   - Bấm vào tên collection → tab **Variables**.
   - Đặt **`base_url`** ở cột Current Value thành server của bạn, ví dụ
     `http://127.0.0.1:8100` (không có dấu `/` ở cuối). **Save** (Ctrl/Cmd-S).
5. Với mỗi request `fuse`, bạn phải tự đính kèm lại file (Postman không lưu nội
   dung file): mở request → **Body → form-data** → ở mỗi dòng `files` bấm
   **Select Files** và chọn một crop biển số.

Sau đó chuyển tới **§3 Các kịch bản kiểm thử**.

---

## 2. Tự tạo request bằng tay (nếu không import)

Tạo một collection mới, thêm biến `base_url = http://127.0.0.1:8100`, rồi thêm các
request bên dưới. URL dùng `{{base_url}}`.

### 2a. Kiểm tra sức khoẻ — `GET {{base_url}}/health`

Không có tham số. Bấm Send. Kỳ vọng `200` và đúng:

```json
{ "status": "ok" }
```

**Đây là bài test đầu tiên — nếu nó lỗi thì mọi thứ khác sẽ không chạy.**

### 2b. Gộp ảnh — `POST {{base_url}}/fuse`

- Phương thức **POST**, URL `{{base_url}}/fuse`.
- Tab **Body** → chọn **form-data**.
- Thêm một dòng: **Key** = `files`, và ở bên phải ô key đổi kiểu từ _Text_ sang
  **File**. Sau đó **Select Files** → chọn một crop.
- Thêm **nhiều dòng `files`** (mỗi dòng kiểu File) cho phần còn lại của burst —
  cần **2+ crop cùng kích thước** của **cùng một** biển số thì gộp mới có tác dụng.
- Tab **Params** → thêm `engine` = `mflpr2` (và tuỳ chọn `scale` = `2`).
- Bấm Send. Bạn nhận lại một **ảnh biển số PNG đã phục dựng** (xem §4 cách xem/lưu).

> Tên trường **bắt buộc là `files`** (số nhiều), mỗi crop một dòng, tất cả đặt
> kiểu File. Mọi crop phải **cùng kích thước pixel** và đúng thứ tự.

---

## 3. Các kịch bản kiểm thử

Kết quả `/fuse` phụ thuộc `engine` và `scale`. Dùng 2+ crop cùng kích thước của
một biển số. Chạy theo thứ tự.

| #   | files            | Query params            | Phản hồi kỳ vọng                                                  |
| --- | ---------------- | ----------------------- | ---------------------------------------------------------------- |
| 1   | 3 crop (96×32)   | `engine=mflpr2`         | `200`, **PNG**, biển số màu đã phục dựng (cùng kích thước input)  |
| 2   | 3 crop (96×32)   | `engine=mflpr2&scale=2` | `200`, **PNG** kích thước **2×** (192×64) — nét hơn               |
| 3   | 3 crop           | `engine=eott`           | `200`, **PNG**, biển số **nhị phân** (đen/trắng), lưới cố định    |
| 4   | 3 crop           | `engine=eott&scale=2`   | `200`, **PNG** — cùng kích thước kịch bản 3 (**eott bỏ qua scale**) |
| 5   | 1 crop           | `engine=mflpr2`         | `200`, **PNG** (một frame, không có gì để gộp — passthrough)      |
| 6   | 3 crop           | `engine=bogus`          | **`400`**, `unknown engine: 'bogus'`                             |
| 7   | file `.txt`/rác  | `engine=mflpr2`         | **`400`**, `no decodable frames`                                 |
| 8   | (không có dòng files) | `engine=mflpr2`    | **`422`**, lỗi validation của FastAPI (thiếu `files`)            |
| 9   | crop **khác kích thước** | `engine=mflpr2` | **`500`**, `mflpr2 failed: ...` (engine cần các frame bằng kích thước) |

> Kịch bản 1 vs 3 là so sánh then chốt: **mflpr2** trả về biển số *màu* đã phục
> dựng và có scale; **eott** trả về biển số *nhị phân* trên lưới cố định và bỏ
> qua `scale`. Chọn theo cái mà OCR phía sau của bạn cần.

---

## 4. Xử lý phản hồi nhị phân (PNG)

Mỗi lần `/fuse` thành công đều trả về ảnh, không phải JSON. Postman hiển thị
trong khung response:

- Tab **Preview** hiển thị ảnh PNG đã gộp.
- Để lưu: bấm **⋯ / Save Response → Save to a file**, đặt đuôi `.png`.

Lỗi (`400`/`422`/`500`) trả về dạng **JSON** trong body — đọc trường `detail` để
biết lý do.

---

## 5. Tham số cho từng request (query)

Truyền ở tab **Params**. Cả hai đều tuỳ chọn.

| Tham số  | Kiểu   | Giá trị             | Ý nghĩa                                                          |
| -------- | ------ | ------------------- | --------------------------------------------------------------- |
| `engine` | string | `mflpr2` \| `eott`  | engine gộp (mặc định `mflpr2`). `eott` xuất ảnh nhị phân.        |
| `scale`  | int    | `1`–`8` (mặc định `1`) | hệ số phóng to — **`mflpr2` áp dụng, `eott` bỏ qua**          |

Không có lựa chọn model/weights và không có cấu hình env phải lo — hai engine cố
định và luôn sẵn sàng khi server đã chạy.

---

## 6. Danh sách kiểm tra khi báo lỗi

Khi thấy bất thường, ghi lại cho dev:

- **Request**: URL đầy đủ kèm `engine`/`scale`, số crop + kích thước pixel.
- **Response**: mã trạng thái, và `detail` JSON (với lỗi) hoặc ảnh PNG đã lưu
  (với vấn đề hình ảnh).
- Output `GET /health` (xác nhận server đang chạy).
- Kỳ vọng vs thực tế (ví dụ "biển số gộp bị mờ", "output eott toàn đen", "scale
  không có tác dụng").

Các kết quả **không phải lỗi** thường gặp:

- `400 unknown engine '...'` → sai chính tả `engine`; dùng `mflpr2` hoặc `eott`.
- `400 no decodable frames` → file đã upload không phải ảnh hợp lệ.
- `500 ... failed` với **crop khác kích thước** → cắt mọi frame về cùng kích
  thước trước; engine register các frame bằng kích thước.
- Output `eott` bỏ qua `scale` và trông **nhị phân** (đen/trắng) → đó là kết quả
  pre-OCR gốc của eott, không phải lỗi. Dùng `mflpr2` để có biển số màu.
- **Một** crop trả về gần như ảnh gốc → một frame thì không có gì để gộp.
