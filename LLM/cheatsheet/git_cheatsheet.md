# Git và GitHub Cheat Sheet

> [!TIP]
> Lưu trữ tài liệu này để tham khảo nhanh các lệnh Git và GitHub thường xuyên sử dụng nhất trong công việc hàng ngày.

## 1. Cấu hình ban đầu (Configuration)

Cài đặt thông tin người dùng cho tất cả các kho lưu trữ cục bộ (local repositories).

```bash
# Đặt tên người dùng
git config --global user.name "Tên của bạn"

# Đặt email liên kết với tài khoản GitHub
git config --global user.email "email@example.com"

# Kiểm tra cấu hình
git config --list
```

## 2. Khởi tạo & Clone Repository

```bash
# Khởi tạo một kho lưu trữ Git mới trong thư mục hiện tại
git init

# Clone một kho lưu trữ từ GitHub về máy
git clone <url-của-repository>
```

## 3. Quản lý thay đổi (Staging & Committing)

```bash
# Kiểm tra trạng thái các file (những file bị thay đổi, file mới, etc.)
git status

# Thêm một file cụ thể vào khu vực chuẩn bị (Staging area)
git add <tên-file>

# Thêm tất cả các file thay đổi vào Staging area
git add .

# Ghi lại các thay đổi vào lịch sử kho lưu trữ (Commit)
git commit -m "Nội dung thông điệp mô tả thay đổi"

# Bỏ qua staging, commit thẳng những file đã được theo dõi (tracked)
git commit -am "Nội dung commit"
```

## 4. Quản lý Nhánh (Branching)

Nhánh giúp bạn làm việc trên các tính năng mới một cách độc lập mà không ảnh hưởng tới code chính.

```bash
# Liệt kê tất cả các nhánh hiện có (nhánh hiện tại có dấu *)
git branch

# Tạo một nhánh mới
git branch <tên-nhánh>

# Chuyển sang một nhánh khác
git checkout <tên-nhánh>
# hoặc (ở các bản Git mới hơn): 
git switch <tên-nhánh>

# Tạo một nhánh mới và chuyển sang nhánh đó ngay lập tức
git checkout -b <tên-nhánh>
# hoặc: 
git switch -c <tên-nhánh>

# Xóa một nhánh (phải đảm bảo không đang ở trên nhánh đó)
git branch -d <tên-nhánh>
```

## 5. Đồng bộ hóa với GitHub (Syncing with Remote)

```bash
# Xem danh sách các remote repositories
git remote -v

# Thêm một remote repository
git remote add origin <url-của-repository>

# Tải về những thay đổi mới nhất từ remote nhưng chưa gộp vào nhánh (Fetch)
git fetch

# Tải về những thay đổi mới nhất và gộp tự động vào nhánh hiện tại (Pull)
git pull origin <tên-nhánh>

# Đẩy các thay đổi ở nhánh hiện tại lên GitHub (Push)
git push origin <tên-nhánh>

# Đẩy nhánh mới tạo ở local lên GitHub lần đầu tiên (tạo upstream)
git push -u origin <tên-nhánh>
```

## 6. Xem Lịch sử & Hủy bỏ thao tác (Logs & Undoing)

> [!CAUTION]
> Cẩn thận khi sử dụng các lệnh undo hoặc reset, đặc biệt là khi làm việc trên những nhánh đã được đẩy lên GitHub (public branches).

```bash
# Xem lịch sử các commits
git log

# Xem lịch sử commits hiển thị gọn gàng (mỗi commit 1 dòng)
git log --oneline

# Xem những thay đổi cụ thể trên file chưa được add
git diff

# Bỏ thay đổi của một file ở Working Directory (đưa về trạng thái commit gần nhất)
git checkout -- <tên-file>
# hoặc: 
git restore <tên-file>

# Đưa file từ Staging area trở về Working Directory (Unstage)
git reset HEAD <tên-file>
# hoặc: 
git restore --staged <tên-file>

# Hoàn tác một commit (tạo ra một commit mới đảo ngược commit cũ)
git revert <commit-id>
```

## 7. Gộp Nhánh (Merging & Rebasing)

```bash
# Gộp một nhánh khác vào nhánh hiện tại
git merge <tên-nhánh-cần-gộp>

# Áp dụng các commit của nhánh hiện tại lên đầu một nhánh gốc (Rebase)
# Lưu ý: Không rebase trên những nhánh public!
git rebase <tên-nhánh-gốc>
```

## 8. Lưu trữ tạm thời (Stashing)

Khi bạn đang làm dở việc nhưng cần chuyển nhánh gấp, stashing giúp bạn lưu lại thay đổi mà chưa cần tạo commit.

```bash
# Lưu trữ tạm thời các thay đổi
git stash

# Liệt kê danh sách các stash hiện có
git stash list

# Lấy lại thay đổi từ stash gần nhất và xóa nó khỏi danh sách
git stash pop

# Lấy lại thay đổi nhưng giữ nguyên stash đó trong danh sách
git stash apply
```
