function softmax(x, axis) result(ret)
    implicit none
    real(kind=8), dimension(:, :), intent(in) :: x
    integer(kind=4), intent(in), optional :: axis
    real(kind=8), dimension(size(x,1), size(x,2)) :: ret
    ! local var
    real(kind=8), dimension(size(x,1), size(x,2)) :: exp_prob
    integer(kind=4) :: axis_
    integer :: i

    if(present(axis))then
        axis_ = axis
    else
        axis_ = 0
    endif
    ! ========================================================
    exp_prob = exp(x)

    if (axis_ == 0) then
        ! follow the python convension
        do i = 1, ubound(exp_prob, 2)
            ret(:, i) = exp_prob(:, i) / sum(exp_prob(:, i))
        end do
    else
        do i = 1, ubound(exp_prob, 1)
            ret(i, :) = exp_prob(i, :) / sum(exp_prob(i, :))
        end do
    end if
end function softmax
